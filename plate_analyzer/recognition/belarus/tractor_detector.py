import numpy as np


class TractorPlateDetector:
    """
    Детектор двухрядного белорусского тракторного номера.

    Формат:

        AB-1
        1234

    Например:

        EK-4
        8765

    Основной признак:
        верхняя строка содержит буквы + код региона,
        нижняя строка содержит 4 цифры.
    """

    def detect(self, detections):
        if not detections:
            return None

        upper_candidates = []
        lower_candidates = []

        for detection in detections:
            text = str(
                detection.get("text", "")
            ).strip().upper()

            box = detection.get("box")

            if not text or box is None:
                continue

            points = np.array(
                box,
                dtype=np.float32,
            )

            if points.size == 0:
                continue

            x1 = float(np.min(points[:, 0]))
            x2 = float(np.max(points[:, 0]))
            y1 = float(np.min(points[:, 1]))
            y2 = float(np.max(points[:, 1]))

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            center_y = (y1 + y2) / 2

            # Нижняя строка:
            # ровно 4 цифры
            if len(text) == 4 and text.isdigit():
                lower_candidates.append({
                    "text": text,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "center_y": center_y,
                })

            # Верхняя строка:
            # 2 буквы + цифра региона
            if (
                len(text) == 3
                and text[:2].isalpha()
                and text[2].isdigit()
            ):
                upper_candidates.append({
                    "text": text,
                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,
                    "width": width,
                    "height": height,
                    "center_y": center_y,
                })

        if not upper_candidates:
            return None

        if not lower_candidates:
            return None

        best_match = None

        for upper in upper_candidates:
            for lower in lower_candidates:

                # Нижняя строка должна находиться ниже верхней.
                if lower["center_y"] <= upper["center_y"]:
                    continue

                vertical_distance = (
                    lower["y1"] - upper["y2"]
                )

                # Слишком большой вертикальный разрыв
                # скорее всего означает два разных объекта.
                if vertical_distance > upper["height"] * 3:
                    continue

                # Строки должны находиться примерно
                # друг под другом.
                upper_center_x = (
                    upper["x1"] + upper["x2"]
                ) / 2

                lower_center_x = (
                    lower["x1"] + lower["x2"]
                ) / 2

                horizontal_distance = abs(
                    upper_center_x - lower_center_x
                )

                max_horizontal_distance = max(
                    upper["width"],
                    lower["width"],
                )

                if (
                    horizontal_distance
                    > max_horizontal_distance
                ):
                    continue

                match = {
                    "detected": True,
                    "upper_text": upper["text"],
                    "lower_text": lower["text"],
                    "text": (
                        upper["text"]
                        + lower["text"]
                    ),
                    "upper_box": upper,
                    "lower_box": lower,
                    "vertical_distance": vertical_distance,
                    "horizontal_distance": (
                        horizontal_distance
                    ),
                }

                if best_match is None:
                    best_match = match
                    continue

                current_distance = (
                    vertical_distance
                    + horizontal_distance
                )

                best_distance = (
                    best_match["vertical_distance"]
                    + best_match["horizontal_distance"]
                )

                if current_distance < best_distance:
                    best_match = match

        return best_match