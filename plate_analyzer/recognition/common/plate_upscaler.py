import cv2


class PlateUpscaler:

    def __init__(
        self,
        min_height=32,
        target_height=96,
        max_scale=4,
    ):
        self.min_height = min_height
        self.target_height = target_height
        self.max_scale = max_scale

    def upscale(self, plate_image):
        if plate_image is None or plate_image.size == 0:
            return {
                "image": None,
                "scale": 1.0,
                "original_height": 0,
                "final_height": 0,
            }

        height, width = plate_image.shape[:2]

        if height >= self.target_height:
            return {
                "image": plate_image,
                "scale": 1.0,
                "original_height": height,
                "final_height": height,
            }

        scale = self.target_height / height

        scale = min(
            scale,
            self.max_scale,
        )

        # Не делаем бессмысленное увеличение.
        if height >= self.min_height:
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_LANCZOS4

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        resized = cv2.resize(
            plate_image,
            (new_width, new_height),
            interpolation=interpolation,
        )

        return {
            "image": resized,
            "scale": round(scale, 2),
            "original_height": height,
            "final_height": new_height,
        }