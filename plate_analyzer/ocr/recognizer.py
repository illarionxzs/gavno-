
import cv2
from rapidocr import RapidOCR


class OCRRecognizer:
    def __init__(self):
        self.ocr = RapidOCR()

    def _run_ocr(self, image):
        result = self.ocr(
            image,
            return_word_box=True,
            return_single_char_box=True,
        )

        detections = []

        if result.txts is None or result.boxes is None:
            return detections

        scores = result.scores or []

        for box, text, score in zip(
            result.boxes,
            result.txts,
            scores,
        ):
            if not text or not text.strip():
                continue

            detections.append({
                "text": text,
                "confidence": float(score),
                "box": box.tolist(),
            })

        return detections

    def _preprocess_for_ocr(self, image):
        # Перевод в оттенки серого
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        # Локальное повышение контраста
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        contrast = clahe.apply(gray)

        # Адаптивная бинаризация
        binary = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        return cv2.cvtColor(
            binary,
            cv2.COLOR_GRAY2BGR,
        )

    def recognize(self, image):
        # Проход 1: исходное изображение
        detections = self._run_ocr(image)

        if detections:
            return detections

        # Проход 2: обработанное изображение
        processed = self._preprocess_for_ocr(image)
        detections = self._run_ocr(processed)

        return detections
