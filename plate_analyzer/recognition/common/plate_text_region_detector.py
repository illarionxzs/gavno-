import cv2
import numpy as np


class PlateTextRegionDetector:

    def __init__(
            self,
            min_char_height_ratio=0.12,
            max_char_height_ratio=0.80,
            vertical_tolerance_ratio=0.30,
            padding_ratio=0.30,
            min_confidence=0.35,
    ):
        self.min_char_height_ratio = (
            min_char_height_ratio
        )

        self.max_char_height_ratio = (
            max_char_height_ratio
        )

        self.vertical_tolerance_ratio = (
            vertical_tolerance_ratio
        )

        self.padding_ratio = padding_ratio
        self.min_confidence = min_confidence

    def detect(self, plate_roi):

        if (
            plate_roi is None
            or plate_roi.size == 0
        ):
            return self._empty_result()

        height, width = plate_roi.shape[:2]

        if height < 20 or width < 40:
            return self._empty_result()

        gray = cv2.cvtColor(
            plate_roi,
            cv2.COLOR_BGR2GRAY,
        )

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0,
        )

        candidates = []

        thresholds = [
            cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY_INV
                + cv2.THRESH_OTSU,
            )[1],

            cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY
                + cv2.THRESH_OTSU,
            )[1],
        ]

        for binary in thresholds:

            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (2, 2),
            )

            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                kernel,
            )

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                x, y, w, h = cv2.boundingRect(
                    contour
                )

                height_ratio = h / height
                width_ratio = w / max(h, 1)

                if not (
                    self.min_char_height_ratio
                    <= height_ratio
                    <= self.max_char_height_ratio
                ):
                    continue

                if width_ratio > 2.5:
                    continue

                if w * h < 10:
                    continue

                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "center_y": y + h / 2,
                    }
                )

        if not candidates:
            return self._empty_result()

        lines = self._group_into_lines(
            candidates,
            height,
        )

        if not lines:
            return self._empty_result()

        best_line = max(
            lines,
            key=lambda line: self._score_line(
                line,
                width,
                height,
            ),
        )

        confidence = self._calculate_confidence(
            best_line,
            width,
            height,
        )

        # -------------------------------------------------
        # Если уверенности недостаточно —
        # НЕ режем изображение вообще.
        # -------------------------------------------------

        if confidence < self.min_confidence:
            return {
                "bbox": (
                    0,
                    0,
                    width,
                    height,
                ),
                "crop": plate_roi.copy(),
                "confidence": round(
                    confidence,
                    3,
                ),
                "characters": len(best_line),
                "full_width": True,
            }

        # -------------------------------------------------
        # ВАЖНО:
        #
        # X НЕ обрезаем вообще.
        #
        # Нам нужно сохранить:
        # - код страны
        # - все буквы
        # - все цифры
        # - края номера
        # -------------------------------------------------

        x1 = 0
        x2 = width

        # По Y определяем положение основной
        # строки символов.

        line_y1 = min(
            item["y"]
            for item in best_line
        )

        line_y2 = max(
            item["y"] + item["h"]
            for item in best_line
        )

        line_height = line_y2 - line_y1

        padding_y = max(
            int(
                line_height
                * self.padding_ratio
            ),
            5,
        )

        y1 = max(
            0,
            line_y1 - padding_y,
        )

        y2 = min(
            height,
            line_y2 + padding_y,
        )

        # Дополнительная защита:
        # не позволяем получить слишком узкий
        # вертикальный crop.

        min_crop_height = int(
            height * 0.35
        )

        if y2 - y1 < min_crop_height:

            center_y = (
                line_y1 + line_y2
            ) // 2

            half_height = (
                min_crop_height // 2
            )

            y1 = max(
                0,
                center_y - half_height,
            )

            y2 = min(
                height,
                center_y + half_height,
            )

        if y2 <= y1:
            return self._empty_result()

        crop = plate_roi[
            y1:y2,
            x1:x2,
        ].copy()

        return {
            "bbox": (
                x1,
                y1,
                x2,
                y2,
            ),
            "crop": crop,
            "confidence": round(
                confidence,
                3,
            ),
            "characters": len(best_line),
            "full_width": True,
        }

    def _group_into_lines(
            self,
            candidates,
            image_height,
    ):

        candidates = sorted(
            candidates,
            key=lambda item: item["center_y"],
        )

        lines = []

        for candidate in candidates:

            added = False

            for line in lines:

                mean_y = np.mean(
                    [
                        item["center_y"]
                        for item in line
                    ]
                )

                mean_height = np.mean(
                    [
                        item["h"]
                        for item in line
                    ]
                )

                tolerance = max(
                    mean_height
                    * self.vertical_tolerance_ratio,
                    image_height * 0.04,
                )

                if abs(
                    candidate["center_y"]
                    - mean_y
                ) <= tolerance:

                    line.append(candidate)

                    added = True

                    break

            if not added:
                lines.append(
                    [candidate]
                )

        return lines

    def _score_line(
            self,
            line,
            image_width,
            image_height,
    ):

        if not line:
            return 0.0

        count_score = min(
            len(line) / 6.0,
            1.0,
        )

        heights = [
            item["h"]
            for item in line
        ]

        median_height = np.median(
            heights
        )

        height_score = min(
            median_height
            / (image_height * 0.45),
            1.0,
        )

        # Нас интересует не горизонтальная
        # длина отдельных символов, а то,
        # насколько много символов найдено
        # примерно на одной высоте.

        y_values = [
            item["center_y"]
            for item in line
        ]

        if len(y_values) > 1:

            spread = (
                max(y_values)
                - min(y_values)
            )

            line_consistency = max(
                0.0,
                1.0
                - spread
                / max(
                    median_height * 1.5,
                    1,
                ),
            )

        else:
            line_consistency = 0.0

        return (
            count_score * 0.45
            + height_score * 0.35
            + line_consistency * 0.20
        )

    def _calculate_confidence(
            self,
            line,
            image_width,
            image_height,
    ):

        return max(
            0.0,
            min(
                self._score_line(
                    line,
                    image_width,
                    image_height,
                ),
                1.0,
            ),
        )

    @staticmethod
    def _empty_result():

        return {
            "bbox": None,
            "crop": None,
            "confidence": 0.0,
            "characters": 0,
            "full_width": False,
        }