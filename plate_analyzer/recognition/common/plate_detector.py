from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PlateDetector:

    def __init__(
            self,
            model_path=None,
            confidence_threshold=0.25,
            image_size=640,
            padding_x=0.15,
            padding_y=0.20,
            merge_overlap_threshold=0.5,
    ):
        if YOLO is None:
            raise ImportError(
                "Не установлен ultralytics. "
                "Выполните: pip install ultralytics"
            )

        if model_path is None:
            project_root = (
                Path(__file__).resolve().parents[2]
            )

            model_path = (
                project_root
                / "models"
                / "license_plate"
                / "license-plate-finetune-v1n.pt"
            )

        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        self.padding_x = padding_x
        self.padding_y = padding_y

        self.merge_overlap_threshold = (
            merge_overlap_threshold
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Не найдены веса детектора:\n"
                f"{self.model_path}"
            )

        self.model = YOLO(
            str(self.model_path)
        )

    def detect(self, image):

        if image is None or image.size == 0:
            return self._empty_result()

        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            imgsz=self.image_size,
            verbose=False,
            device="cpu",
        )

        if not results:
            return self._empty_result()

        result = results[0]

        if result.boxes is None:
            return self._empty_result()

        if len(result.boxes) == 0:
            return self._empty_result()

        height, width = image.shape[:2]

        raw_detections = []

        # =====================================================
        # Получаем bbox от YOLO
        # =====================================================

        for index in range(len(result.boxes)):

            confidence = float(
                result.boxes.conf[index].item()
            )

            xyxy = (
                result
                .boxes
                .xyxy[index]
                .cpu()
                .numpy()
            )

            x1, y1, x2, y2 = [
                int(round(value))
                for value in xyxy
            ]

            if x2 <= x1 or y2 <= y1:
                continue

            raw_detections.append(
                {
                    "detection_bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "confidence": round(
                        confidence,
                        3,
                    ),
                }
            )

        if not raw_detections:
            return self._empty_result()

        # =====================================================
        # Объединяем сильно пересекающиеся bbox
        # =====================================================

        merged_detections = (
            self._merge_overlapping_detections(
                raw_detections
            )
        )

        detections = []

        # =====================================================
        # Padding + crop
        # =====================================================

        for detection in merged_detections:

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            confidence = detection["confidence"]

            # -------------------------
            # Размер bbox
            # -------------------------

            box_width = x2 - x1
            box_height = y2 - y1

            # -------------------------
            # Padding
            # -------------------------

            pad_x = max(
                int(box_width * self.padding_x),
                5,
            )

            pad_y = max(
                int(box_height * self.padding_y),
                5,
            )

            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            # -------------------------
            # Clamp
            # -------------------------

            x1 = max(
                0,
                min(x1, width - 1),
            )

            y1 = max(
                0,
                min(y1, height - 1),
            )

            x2 = max(
                1,
                min(x2, width),
            )

            y2 = max(
                1,
                min(y2, height),
            )

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[
                y1:y2,
                x1:x2,
            ].copy()

            detections.append(
                {
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                    "detection_bbox": detection[
                        "detection_bbox"
                    ],
                    "merged_bbox": detection[
                        "merged_bbox"
                    ],
                    "confidence": confidence,
                    "crop": crop,
                }
            )

        if not detections:
            return self._empty_result()

        # =====================================================
        # Лучший кандидат
        # =====================================================

        best = max(
            detections,
            key=lambda item: item["confidence"],
        )

        return {
            "bbox": best["bbox"],
            "confidence": best["confidence"],
            "crop": best["crop"],
            "detections": detections,
        }

    # =========================================================
    # Merge overlapping detections
    # =========================================================

    def _merge_overlapping_detections(
        self,
        detections,
    ):

        if len(detections) <= 1:
            detection = detections[0]

            bbox = detection["detection_bbox"]

            return [
                {
                    "bbox": bbox,
                    "detection_bbox": bbox,
                    "merged_bbox": bbox,
                    "confidence": detection[
                        "confidence"
                    ],
                }
            ]

        used = set()
        merged = []

        for i, first in enumerate(detections):

            if i in used:
                continue

            first_bbox = first["detection_bbox"]

            x1, y1, x2, y2 = first_bbox

            confidence = first["confidence"]

            group = [
                first_bbox
            ]

            used.add(i)

            changed = True

            # -------------------------------------------------
            # Повторяем поиск, потому что после объединения
            # новая область может пересекаться с третьим bbox.
            # -------------------------------------------------

            while changed:

                changed = False

                for j, second in enumerate(
                    detections
                ):

                    if j in used:
                        continue

                    second_bbox = (
                        second["detection_bbox"]
                    )

                    if self._boxes_overlap(
                        (x1, y1, x2, y2),
                        second_bbox,
                    ):

                        sx1, sy1, sx2, sy2 = (
                            second_bbox
                        )

                        x1 = min(
                            x1,
                            sx1,
                        )

                        y1 = min(
                            y1,
                            sy1,
                        )

                        x2 = max(
                            x2,
                            sx2,
                        )

                        y2 = max(
                            y2,
                            sy2,
                        )

                        confidence = max(
                            confidence,
                            second["confidence"],
                        )

                        group.append(
                            second_bbox
                        )

                        used.add(j)

                        changed = True

            merged_bbox = (
                x1,
                y1,
                x2,
                y2,
            )

            # -------------------------------------------------
            # Если было несколько bbox — объединённый bbox.
            # Если один — остаётся обычным.
            # -------------------------------------------------

            if len(group) > 1:

                original_bbox = group[0]

                merged.append(
                    {
                        "bbox": merged_bbox,
                        "detection_bbox": (
                            original_bbox
                        ),
                        "merged_bbox": merged_bbox,
                        "confidence": round(
                            confidence,
                            3,
                        ),
                    }
                )

            else:

                merged.append(
                    {
                        "bbox": merged_bbox,
                        "detection_bbox": (
                            group[0]
                        ),
                        "merged_bbox": merged_bbox,
                        "confidence": round(
                            confidence,
                            3,
                        ),
                    }
                )

        return merged

    # =========================================================
    # Проверка пересечения bbox
    # =========================================================

    def _boxes_overlap(
        self,
        first,
        second,
    ):

        x1, y1, x2, y2 = first
        sx1, sy1, sx2, sy2 = second

        intersection_x1 = max(
            x1,
            sx1,
        )

        intersection_y1 = max(
            y1,
            sy1,
        )

        intersection_x2 = min(
            x2,
            sx2,
        )

        intersection_y2 = min(
            y2,
            sy2,
        )

        if (
            intersection_x2 <= intersection_x1
            or intersection_y2 <= intersection_y1
        ):
            return False

        intersection_area = (
            (
                intersection_x2
                - intersection_x1
            )
            *
            (
                intersection_y2
                - intersection_y1
            )
        )

        first_area = (
            (x2 - x1)
            *
            (y2 - y1)
        )

        second_area = (
            (sx2 - sx1)
            *
            (sy2 - sy1)
        )

        if first_area <= 0 or second_area <= 0:
            return False

        smaller_area = min(
            first_area,
            second_area,
        )

        overlap_ratio = (
            intersection_area
            / smaller_area
        )

        return (
            overlap_ratio
            >= self.merge_overlap_threshold
        )

    # =========================================================
    # Empty result
    # =========================================================

    @staticmethod
    def _empty_result():

        return {
            "bbox": None,
            "confidence": 0.0,
            "crop": None,
            "detections": [],
        }