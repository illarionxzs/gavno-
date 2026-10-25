
from pathlib import Path

import cv2

from recognition.common.plate_background_detector import PlateBackgroundDetector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = PROJECT_ROOT / "images" / "test64_plate.jpg"


def main():
    print(f"Ищем изображение:")
    print(IMAGE_PATH)
    print()

    if not IMAGE_PATH.exists():
        print("ОШИБКА: файл не найден!")
        print(f"Путь: {IMAGE_PATH}")
        return

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("ОШИБКА: файл существует, но OpenCV не смог его прочитать!")
        print(f"Путь: {IMAGE_PATH}")
        return

    print("Изображение успешно загружено.")
    print(f"Размер: {image.shape[1]} x {image.shape[0]}")
    print()

    detector = PlateBackgroundDetector()

    result = detector.detect(image)

    print("BACKGROUND:")
    print(result)


if __name__ == "__main__":
    main()



