import os
import time

# Ограничиваем количество потоков для математических библиотек
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch

torch.set_num_threads(6)

from pathlib import Path
import re

import cv2
from rapidocr import RapidOCR

from ocr.recognizer import OCRRecognizer
from recognition.belarus import BelarusClassifier
from recognition.russia.classifier import RussiaClassifier
from recognition.belarus.temporary_detector import TemporaryPlateDetector
from recognition.common.plate_background_detector import PlateBackgroundDetector
from recognition.common.plate_detector import PlateDetector
from recognition.normalizer import normalize_detections
from storage.database import PlateDatabase

# Корневая папка проекта — каталог, в котором находится main.py.
PROJECT_ROOT = Path(__file__).resolve().parent

# Исходные фотографии и результаты.
IMAGES_DIR = PROJECT_ROOT / "test_images"
CROPS_DIR = PROJECT_ROOT / "results" / "plate_detection"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

# =========================================================
# БЕЛОРУССКИЕ МАСКИ ДЛЯ СТРОГОЙ ВАЛИДАЦИИ
# =========================================================
BY_PLATE_MASKS = [
    # 1. СТАРЫЕ ФОРМАТЫ
    r"^\d{4}[ABCEHIKMOPTX]{2}$",
    r"^\d{4}[ABCEHIKMOPTX]{3}$",
    r"^[ABCEHIKMOPTX]{2}\d{4}$",
    r"^[ABCEHIKMOPTX]\d{4}[ABCEHIKMOPTX]$",
    r"^[ABCEHIKMOPTX]\d{5}$",

    # 2. Спецформаты
    r"^\d{4}(BN|BT|GS|GK|MO|MG|MH|MI|TK|EK|ME)$",
    r"^[1-8](TAX|TBX|TEX)\d{4}$",
    r"^[1-8][ABCEHIKMOPTX]{3}\d{4}$",
    r"^(CD|CC)\d{4}[1-8]$",

    # 3. Современные стандартные
    r"^\d{4}[ABCEHIKMOPTX]{2}[1-8]$",
    r"^[ABCEHIKMOPTX]{2}\d{4}[1-8]$",
    r"^[ABCEHIKMOPTX]\d{4}[ABCEHIKMOPTX][1-8]$",
    r"^E\d{3}[ABCEHIKMOPTX]{2}[1-8]$",
    r"^[ABCEHIKMOPTX]{2}E\d{3}[1-8]$",
    r"^[ABCEHIKMOPTX]{2}\d{4}[ \-]?0$",
]

# =========================================================
# МЕЖДУНАРОДНЫЕ ПРЕФИКСЫ (РЕШЕНИЕ КОЛЛИЗИЙ С ОДИНАКОВЫМИ МАСКАМИ)
# =========================================================
COUNTRY_PREFIXES = { #попытка разделить литовские и новые молдавские номера формата ABC123 (неудачная)
    "MD": {"code": "MD", "name": "Молдова"},
    "KZ": {"code": "KZ", "name": "Казахстан"},
    "BY": {"code": "BY", "name": "Беларусь"},
    "RU": {"code": "RU", "name": "Россия"},
    "RUS": {"code": "RUS", "name": "Россия"},
    "LT": {"code": "LT", "name": "Литва"},
}


def parse_plate_with_prefix(raw_text: str): #попытка разделить литовские и новые молдавские номера формата ABC123 (неудачная)
    """
    Пытается найти код страны в начале строки (например, 'MDABC123' -> 'MD' и 'ABC123').
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())

    # Проверяем длинные префиксы первыми (например, 'RUS' раньше, чем 'RU')
    for prefix in sorted(COUNTRY_PREFIXES.keys(), key=len, reverse=True):
        if cleaned.startswith(prefix):
            pure_number = cleaned[len(prefix):]
            country_info = COUNTRY_PREFIXES[prefix]
            return {
                "country_code": country_info["code"],
                "country": country_info["name"],
                "number": pure_number,
                "has_explicit_prefix": True
            }

    return {
        "country_code": None,
        "country": None,
        "number": cleaned,
        "has_explicit_prefix": False
    }


def detect_country_from_blue_strip(plate_crop, ocr_instance) -> str: #попытка разделить литовские и новые молдавские номера формата ABC123 (неудачная)
    """
    Вырезает левую полосу пластины (синий блок с кодом страны)
    и отдельно распознает текст на ней для разделения одинаковых масок (MD vs LT).
    """
    if plate_crop is None or plate_crop.size == 0:
        return None

    h, w = plate_crop.shape[:2]
    # Левая синяя полоса обычно занимает первые 15-20% ширины пластины
    strip_width = int(w * 0.18)
    left_strip = plate_crop[:, :strip_width]

    # Увеличиваем полосу для четкого распознавания мелкого шрифта (MD, LT)
    enlarged_strip = cv2.resize(left_strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Запускаем OCR локально на левом крае
    strip_detections = ocr_instance._run_ocr(enlarged_strip)
    if not strip_detections:
        strip_detections = ocr_instance._run_ocr(left_strip)

    for det in strip_detections:
        text = clean_and_normalize_ocr_text(det.get("text", ""))
        for prefix in ["MD", "LT", "KZ", "BY", "RUS", "RU"]:
            if prefix in text:
                return prefix

    return None

# =========================================================
# ЛЕГКИЕ СТРУКТУРНЫЕ ШАБЛОНЫ (ПЛАН Б) сюда надо добавлять маски номеров разных стран, чтобы они в статистике так же учитывались. ещё нужно добавить код страны в словарь COUNTRY_CODES в файле normalizer.py
# =========================================================

INTERNATIONAL_PATTERNS = {
    "MD": {
        "country": "Молдова",
        "patterns": [
            r"^[A-Z]{3}\d{3}$",           # Новые молдавские: 3 буквы + 3 цифры
            r"^[A-Z]{1,2}[A-Z]{2}\d{3}$", # Старые молдавские
        ]
    },
    "LT": {
        "country": "Литва",
        "patterns": [
            r"^[A-Z]{3}\d{3}$",           # Стандартные литовские: 3 буквы + 3 цифры (например, ABC 123)
            r"^[A-Z]{2}\d{3}$",           # Прицепы и спецформаты
        ]
    },
    "KZ": {
        "country": "Казахстан",
        "patterns": [
            r"^\d{3}[A-Z]{2,3}\d{2}$",   # 3 цифры + 2 или 3 буквы + 2 цифры региона
            r"^[A-Z]\d{3}[A-Z]{3}$",     # Старые казахские форматы
        ]
    }
}


def clean_and_normalize_ocr_text(text: str) -> str:
    """Очищает OCR-текст и заменяет частые ошибки распознавания."""
    if not text:
        return ""
    text = text.upper()

    cyr_to_lat = {
        "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H",
        "І": "I", "К": "K", "М": "M", "О": "O", "Р": "P",
        "Т": "T", "Х": "X", "Б": "B",
    }
    for cyr, lat in cyr_to_lat.items():
        text = text.replace(cyr, lat)

    fixes = {"Z": "2", "Q": "O", "D": "0", "F": "E"}
    for bad, good in fixes.items():
        text = text.replace(bad, good)

    return re.sub(r"[^A-Z0-9\-]", "", text)


def reconstruct_two_line_plate(word_results) -> str:
    """Собирает двухстрочный номер по посимвольным координатам."""
    if not word_results:
        return ""

    all_chars = []
    for block in word_results:
        if not isinstance(block, (list, tuple)):
            continue
        for item in block:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            char, score, box = item
            if not box or len(box) != 4:
                continue

            ys = [point[1] for point in box]
            xs = [point[0] for point in box]
            all_chars.append({
                "char": str(char),
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "score": float(score),
            })

    if not all_chars:
        return ""

    y_coords = [c["y"] for c in all_chars]
    y_middle = (min(y_coords) + max(y_coords)) / 2.0

    top_line = sorted([c for c in all_chars if c["y"] < y_middle], key=lambda c: c["x"])
    bottom_line = sorted([c for c in all_chars if c["y"] >= y_middle], key=lambda c: c["x"])

    return "".join(c["char"] for c in top_line) + "".join(c["char"] for c in bottom_line)


def find_best_valid_detection(all_variants):
    """Проверяет все источники OCR по списку масок."""
    for variant in all_variants:
        if not variant:
            continue

        for item in variant:
            raw_text = item.get("text", "")
            cleaned = clean_and_normalize_ocr_text(raw_text)

            for pattern in BY_PLATE_MASKS:
                if re.fullmatch(pattern, cleaned.replace("-", "")):
                    print(f"[VALIDATOR] Найден валидный номер по маске: {cleaned}")
                    item["text"] = cleaned
                    return [item]

        combined_raw = "".join(str(d.get("text", "")) for d in variant)
        combined_cleaned = clean_and_normalize_ocr_text(combined_raw)
        for pattern in BY_PLATE_MASKS:
            if re.fullmatch(pattern, combined_cleaned.replace("-", "")):
                print(f"[VALIDATOR] Найдена валидная склейка по маске: {combined_cleaned}")
                if variant:
                    variant[0]["text"] = combined_cleaned
                    return [variant[0]]

    return None


def print_general_report(db: PlateDatabase):
    stats = db.get_statistics()

    print("\n" + "=" * 55)
    print(f"📊 ОБЩАЯ СТАТИСТИКА ПО БАЗЕ ДАННЫХ")
    print(f"Всего уникальных машин: {stats['total_unique_vehicles']}")
    print("=" * 55)

    print("\n🌍 По странам:")
    for country_info in stats["countries"]:
        print(f"  • {country_info['country']} ({country_info['country_code']}): {country_info['count']} шт.")

    print("\n📍 Распределение по регионам (РФ и Беларусь):")

    filtered_regions = [
        r for r in stats["regions"]
        if r.get('country_code') in ("BY", "RU")
    ]

    if not filtered_regions:
        print("  (Нет данных по регионам)")
    else:
        for reg in filtered_regions:
            c_code = reg.get('country_code', 'N/A')
            r_code = reg.get('region_code') or "N/A"
            r_name = reg.get('region') or "Не указан"
            count = reg.get('count', 0)
            print(f"  [{c_code}] Регион {r_code} ({r_name}): {count} шт.")

    print("=" * 55)


# =========================================================
# MAIN
# =========================================================

def main():
    db = PlateDatabase()

    plate_detector = PlateDetector()
    ocr = OCRRecognizer()
    debug_ocr = RapidOCR()
    background_detector = PlateBackgroundDetector()
    temporary_detector = TemporaryPlateDetector()
    belarus_classifier = BelarusClassifier()
    russia_classifier = RussiaClassifier()

    if not IMAGES_DIR.exists():
        print(f"Папка с изображениями не найдена: {IMAGES_DIR}")
        return

    image_paths = sorted(
        path
        for path in IMAGES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Найдено фотографий: {len(image_paths)}")

    processed_count = 0
    skipped_count = 0
    error_count = 0

    for image_path in image_paths:
        source_path = image_path.relative_to(PROJECT_ROOT).as_posix()

        print("\n" + "=" * 65)
        print(f"Фотография: {source_path}")

        if db.is_file_processed(source_path):
            print("Уже обработана — пропускаем.")
            skipped_count += 1
            continue

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError("Не удалось открыть изображение")

            print(f"Размер исходного фото: {image.shape[1]} x {image.shape[0]}")

            # -------------------------------------------------
            # DETECTION
            # -------------------------------------------------
            plate_result = plate_detector.detect(image)
            plate_crop = plate_result.get("crop")

            if plate_crop is None or plate_crop.size == 0:
                print("Номер не найден.")
                db.mark_file_processed(source_path, status="done")
                processed_count += 1
                continue

            relative_path = image_path.relative_to(IMAGES_DIR)
            crop_dir = CROPS_DIR / relative_path.parent
            crop_dir.mkdir(parents=True, exist_ok=True)

            crop_path = crop_dir / (
                f"{relative_path.stem}__{relative_path.suffix[1:].lower()}_plate.jpg"
            )
            cv2.imwrite(str(crop_path), plate_crop)
            print(f"Кроп сохранён: {crop_path.relative_to(PROJECT_ROOT)}")

            enlarged = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

            # -------------------------------------------------
            # PLATE COLOR
            # -------------------------------------------------
            background_result = background_detector.detect(plate_crop)
            plate_color = background_result.get("color")

            print("\n=== PLATE BACKGROUND ===")
            print(f"Цвет: {plate_color!r}")

            # -------------------------------------------------
            # OCR VARIANTS & VALIDATION
            # -------------------------------------------------
            original_detections = ocr.recognize(image)
            crop_detections = ocr.recognize(plate_crop)
            upscaled_detections = ocr._run_ocr(enlarged)
            processed_crop = ocr._preprocess_for_ocr(enlarged)
            processed_detections = ocr._run_ocr(processed_crop)

            debug_result = debug_ocr(
                plate_crop,
                return_word_box=True,
                return_single_char_box=True,
            )
            word_results = getattr(debug_result, "word_results", None)
            two_line_text = reconstruct_two_line_plate(word_results)

            two_line_variant = (
                [{
                    "text": two_line_text,
                    "confidence": 0.85,
                    "box": [[0, 0], [0, 0], [0, 0], [0, 0]],
                }]
                if two_line_text
                else []
            )

            all_variants = [
                two_line_variant,
                crop_detections,
                upscaled_detections,
                processed_detections,
                original_detections,
            ]

            valid_detections = find_best_valid_detection(all_variants)
            detections = valid_detections if valid_detections else crop_detections

            print("\n=== OCR ===")
            for detection in detections:
                print(f"Текст: {detection.get('text', 'N/A')}, уверенность: {detection.get('confidence', 0.0):.2f}")

            # -------------------------------------------------
            # TEMPORARY PLATE
            # -------------------------------------------------
            is_clearly_standard = any(
                re.fullmatch(r".*[A-Z]{1,2}\d*[1-8]", str(d.get("text", "")).replace("-", ""))
                for d in detections
            ) if detections else False

            if is_clearly_standard:
                temporary_result = None
            else:
                temporary_result = temporary_detector.detect(
                    image=plate_crop,
                    detections=detections,
                    word_results=word_results,
                )

            print("\n=== TEMPORARY PLATE DETECTOR ===")
            print(f"TEMPORARY: {bool(temporary_result)}")

            # -------------------------------------------------
            # NORMALIZER & ROUTING
            # -------------------------------------------------
            normalized = normalize_detections(detections)

            print("\n=== NORMALIZER ===")
            if normalized is None:
                print("Не удалось распознать номер.")
                db.mark_file_processed(source_path, status="done")
                processed_count += 1
                continue

            raw_number = normalized["number"]
            print(f"Нормализованный текст: {raw_number}")

            result = None

            # ЭТАП 1: Проверка на явный префикс страны
            parsed_info = parse_plate_with_prefix(raw_number) #попытка разделить литовские и новые молдавские номера формата ABC123 (неудачная)

            if parsed_info["has_explicit_prefix"]:
                country_code = parsed_info["country_code"]
                number = parsed_info["number"]
                result = {
                    "number": number,
                    "country_code": country_code,
                    "country": parsed_info["country"],
                    "type": "standard",
                    "region_code": None,
                    "region": f"Страна: {parsed_info['country']} (по префиксу)"
                }
                print(f"[PREFIX DETECTOR] Обнаружен явный код страны: {country_code}, номер: {number}")
            else:
                # ЭТАП 2: Эвристика по количеству блоков OCR (Разделение Молдовы и Литвы) #попытка разделить литовские и новые молдавские номера формата ABC123 (неудачная)
                matched_foreign = False
                valid_raw_texts = [d.get("text", "") for d in detections if d.get("text")]

                if len(valid_raw_texts) >= 2:
                    combined_check = clean_and_normalize_ocr_text("".join(valid_raw_texts))
                    if re.fullmatch(r"^[A-Z]{3}\d{3}$", combined_check):
                        result = {
                            "number": combined_check,
                            "country_code": "MD",
                            "country": "Молдова",
                            "type": "standard",
                            "region_code": None,
                            "region": "Страна: Молдова (по разделителю блоков)"
                        }
                        matched_foreign = True
                        print("[HEURISTIC] Обнаружен разделитель блоков -> Определена Молдова")

                # ЭТАП 3: Проверка по остальным шаблонам (Казахстан и др.)
                if not matched_foreign:
                    for code, info in INTERNATIONAL_PATTERNS.items():
                        if any(re.fullmatch(pattern, raw_number) for pattern in info["patterns"]):
                            result = {
                                "number": raw_number,
                                "country_code": code,
                                "country": info["country"],
                                "type": "standard",
                                "region_code": None,
                                "region": f"Страна: {info['country']}"
                            }
                            matched_foreign = True
                            break

                # ЭТАП 4: Если не иностранный — отдаем в классификаторы РФ и РБ
                if not matched_foreign:
                    result = belarus_classifier.classify(number=raw_number)
                    if not result or result.get("type") == "unknown":
                        result = russia_classifier.classify(number=raw_number, country_code="RU")

            if not result or result.get("type") == "unknown":
                print("Не удалось определить страну и тип номера.")
                db.mark_file_processed(source_path, status="done")
                processed_count += 1
                continue

            print(f"\n=== CLASSIFICATION RESULT ===")
            print(f"Номер: {result['number']}")
            print(f"Страна: {result.get('country', 'Не указана')}")
            print(f"Тип: {result['type']}")
            print(f"Регион: {result.get('region', 'Не указан')}")

            # -------------------------------------------------
            # DATABASE
            # -------------------------------------------------
            saved = db.add_vehicle(result=result, image_path=source_path)
            if saved is not None:
                vehicle_id, is_new = saved
                print(f"{'Новый автомобиль сохранён' if is_new else 'Автомобиль уже известен'}: ID={vehicle_id}")

            db.mark_file_processed(source_path, status="done")
            processed_count += 1

        except Exception as exc:
            error_count += 1
            print(f"Ошибка обработки {source_path}: {exc}")
            try:
                db.mark_file_processed(source_path, status="error", error=str(exc))
            except Exception as db_exc:
                print(f"Не удалось записать ошибку в БД: {db_exc}")

    print("\n" + "=" * 65)
    print("=== ИТОГИ ПАКЕТНОЙ ОБРАБОТКИ ===")
    print(f"Всего найдено: {len(image_paths)}")
    print(f"Обработано: {processed_count}")
    print(f"Пропущено ранее обработанных: {skipped_count}")
    print(f"Ошибок: {error_count}")

    # Вывод общей универсальной статистики
    print_general_report(db)

    time.sleep(0.03)


if __name__ == "__main__":
    main()