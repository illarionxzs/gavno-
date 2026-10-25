
import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "plates.db"
)


class PlateDatabase:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._create_tables()

    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _create_tables(self):
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    country_code TEXT NOT NULL,
                    number TEXT NOT NULL,
                    country TEXT NOT NULL,
                    vehicle_type TEXT,
                    region_code TEXT,
                    region TEXT,
                    first_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(country_code, number)
                );

                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    observed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ocr_corrected INTEGER NOT NULL DEFAULT 0,
                    ocr_original TEXT,
                    FOREIGN KEY(vehicle_id)
                        REFERENCES vehicles(id),
                    UNIQUE(vehicle_id, image_path)
                );

                CREATE TABLE IF NOT EXISTS processed_files (
                    image_path TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    error TEXT,
                    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    @staticmethod
    def _normalize_number(number):
        return "".join(str(number).upper().split())

    def add_vehicle(self, result, image_path):
        """
        Сохраняет распознанный автомобиль и наблюдение.

        Возвращает:
            (vehicle_id, is_new)
        """
        if not result or result.get("type") == "unknown":
            return None

        country_code = result.get("country_code")
        country = result.get("country")
        number = self._normalize_number(
            result.get("number", "")
        )

        if not country_code or not country or not number:
            return None

        image_path = str(Path(image_path).resolve())

        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT id
                FROM vehicles
                WHERE country_code = ? AND number = ?
                """,
                (country_code, number),
            ).fetchone()

            is_new = existing is None

            connection.execute(
                """
                INSERT INTO vehicles (
                    country_code,
                    number,
                    country,
                    vehicle_type,
                    region_code,
                    region
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(country_code, number)
                DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP
                """,
                (
                    country_code,
                    number,
                    country,
                    result.get("type"),
                    result.get("region_code"),
                    result.get("region"),
                ),
            )

            vehicle = connection.execute(
                """
                SELECT id
                FROM vehicles
                WHERE country_code = ? AND number = ?
                """,
                (country_code, number),
            ).fetchone()

            vehicle_id = vehicle["id"]

            connection.execute(
                """
                INSERT INTO observations (
                    vehicle_id,
                    image_path,
                    ocr_corrected,
                    ocr_original
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(vehicle_id, image_path)
                DO NOTHING
                """,
                (
                    vehicle_id,
                    image_path,
                    int(bool(result.get("ocr_corrected"))),
                    result.get("ocr_original"),
                ),
            )

        return vehicle_id, is_new

    def mark_file_processed(self, image_path, status="done", error=None):
        image_path = str(Path(image_path).resolve())

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO processed_files (
                    image_path, status, error
                )
                VALUES (?, ?, ?)
                ON CONFLICT(image_path)
                DO UPDATE SET
                    status = excluded.status,
                    error = excluded.error,
                    processed_at = CURRENT_TIMESTAMP
                """,
                (image_path, status, error),
            )

    def is_file_processed(self, image_path):
        image_path = str(Path(image_path).resolve())

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status
                FROM processed_files
                WHERE image_path = ?
                """,
                (image_path,),
            ).fetchone()

        return row is not None and row["status"] == "done"

    def get_statistics(self):
        with self._connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS count FROM vehicles"
            ).fetchone()["count"]

            countries = connection.execute(
                """
                SELECT country, country_code, COUNT(*) AS count
                FROM vehicles
                GROUP BY country, country_code
                ORDER BY count DESC, country
                """
            ).fetchall()

            regions = connection.execute(
                """
                SELECT country, country_code, region, COUNT(*) AS count
                FROM vehicles
                WHERE country_code IN ('BY', 'RU')
                  AND region IS NOT NULL
                GROUP BY country, country_code, region
                ORDER BY country, count DESC, region
                """
            ).fetchall()

        return {
            "total_unique_vehicles": total,
            "countries": [dict(row) for row in countries],
            "regions": [dict(row) for row in regions],
        }

    def get_belarus_regions_statistics(self):
        """
        Возвращает детализированную статистику по Беларуси:
        - Распределение по регионам (0-8)
        - Количество старых номеров (TYPE_OLD / 'old')
        - Количество номеров временного ввоза (TYPE_TEMPORARY / 'temporary')
        - Общее количество белорусских номеров
        """
        with self._connect() as connection:
            # 1. Общее количество номеров Беларуси
            total_by = connection.execute(
                "SELECT COUNT(*) AS count FROM vehicles WHERE country_code = 'BY'"
            ).fetchone()["count"]

            # 2. Статистика по кодам регионов
            region_rows = connection.execute(
                """
                SELECT region_code, COUNT(*) AS count
                FROM vehicles
                WHERE country_code = 'BY' AND region_code IS NOT NULL
                GROUP BY region_code
                """
            ).fetchall()
            regions = {str(row["region_code"]): row["count"] for row in region_rows}

            # 3. Старые номера
            old_count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM vehicles
                WHERE country_code = 'BY' AND vehicle_type = 'old'
                """
            ).fetchone()["count"]

            # 4. Номера временного ввоза
            temp_count = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM vehicles
                WHERE country_code = 'BY' AND vehicle_type = 'temporary'
                """
            ).fetchone()["count"]

        return {
            "total": total_by,
            "regions": regions,
            "old_count": old_count,
            "temporary_count": temp_count,
        }