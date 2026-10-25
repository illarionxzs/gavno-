import cv2
import numpy as np


class PlateBackgroundDetector:

    def detect(self, plate_image):
        """
        Определяет основной цвет фона номерного знака.

        Parameters
        ----------
        plate_image : numpy.ndarray
            Изображение номера в формате OpenCV (BGR).

        Returns
        -------
        dict
            Например:
            {
                "color": "red",
                "confidence": 0.91,
            }
        """

        if plate_image is None or plate_image.size == 0:
            return {
                "color": "unknown",
                "confidence": 0.0,
            }

        # --------------------------------------------------
        # 1. Немного уменьшаем влияние краёв номера
        # --------------------------------------------------

        height, width = plate_image.shape[:2]

        if height < 5 or width < 5:
            return {
                "color": "unknown",
                "confidence": 0.0,
            }

        margin_x = int(width * 0.10)
        margin_y = int(height * 0.15)

        if margin_x * 2 >= width:
            margin_x = 0

        if margin_y * 2 >= height:
            margin_y = 0

        roi = plate_image[
            margin_y:height - margin_y,
            margin_x:width - margin_x,
        ]

        if roi.size == 0:
            roi = plate_image

        # --------------------------------------------------
        # 2. Переводим BGR → HSV
        # --------------------------------------------------

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --------------------------------------------------
        # 3. Маски красного цвета
        #
        # В HSV красный находится около 0°
        # и около 180°.
        #
        # Поэтому нужны две области.
        # --------------------------------------------------

        lower_red_1 = np.array([0, 70, 40])
        upper_red_1 = np.array([10, 255, 255])

        lower_red_2 = np.array([170, 70, 40])
        upper_red_2 = np.array([179, 255, 255])

        mask_red_1 = cv2.inRange(
            hsv,
            lower_red_1,
            upper_red_1,
        )

        mask_red_2 = cv2.inRange(
            hsv,
            lower_red_2,
            upper_red_2,
        )

        red_mask = cv2.bitwise_or(
            mask_red_1,
            mask_red_2,
        )

        # --------------------------------------------------
        # 4. Убираем мелкий шум
        # --------------------------------------------------

        kernel = np.ones((3, 3), np.uint8)

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        # --------------------------------------------------
        # 5. Считаем долю красных пикселей
        # --------------------------------------------------

        total_pixels = red_mask.size

        red_pixels = cv2.countNonZero(red_mask)

        red_ratio = red_pixels / total_pixels

        # --------------------------------------------------
        # 6. Решение
        # --------------------------------------------------
        #
        # Важно:
        # мы не требуем 100% красного.
        #
        # На номере есть:
        # - цифры
        # - буквы
        # - чёрные элементы
        # - блики
        # - тени
        #
        # Поэтому используем порог.
        # --------------------------------------------------

        if red_ratio >= 0.25:

            confidence = min(
                1.0,
                red_ratio / 0.60,
            )

            return {
                "color": "red",
                "confidence": round(confidence, 3),
            }

        # --------------------------------------------------
        # 7. Белый фон
        #
        # Высокая яркость + низкая насыщенность.
        # --------------------------------------------------

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        white_mask = (
            (saturation < 50) &
            (value > 140)
        )

        white_ratio = (
            np.count_nonzero(white_mask)
            / total_pixels
        )

        if white_ratio >= 0.40:

            confidence = min(
                1.0,
                white_ratio / 0.80,
            )

            return {
                "color": "white",
                "confidence": round(confidence, 3),
            }

        # --------------------------------------------------
        # 8. Ничего уверенно определить не удалось
        # --------------------------------------------------

        return {
            "color": "unknown",
            "confidence": 0.0,
        }