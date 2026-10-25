from pathlib import Path

import cv2

from recognition.common.plate_detector import PlateDetector
from recognition.common.plate_text_region_detector import (
    PlateTextRegionDetector,
)


def main():

    project_root = Path(__file__).resolve().parents[2]

    images_dir = project_root / "images"

    output_dir = (
        project_root
        / "results"
        / "plate_text_region"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plate_detector = PlateDetector()

    text_detector = PlateTextRegionDetector()

    image_paths = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }
        ]
    )

    if not image_paths:
        print(
            "В папке images не найдено изображений."
        )
        return

    print(
        f"Найдено изображений: {len(image_paths)}"
    )
    print()

    for image_path in image_paths:

        print(
            f"Обрабатывается: "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print("  Ошибка чтения изображения")
            print()
            continue

        # -------------------------------------------------
        # 1. Ищем физическую область номерного знака
        # -------------------------------------------------

        plate_result = plate_detector.detect(
            image
        )

        plate_crop = plate_result["crop"]

        if plate_crop is None:
            print(
                "  Номерной знак не найден"
            )
            print()
            continue

        # -------------------------------------------------
        # 2. Ищем основную строку символов
        # -------------------------------------------------

        text_result = text_detector.detect(
            plate_crop
        )

        text_crop = text_result["crop"]

        if text_crop is None:
            print(
                "  Основная строка не найдена"
            )
            print()
            continue

        bbox = text_result["bbox"]

        confidence = (
            text_result["confidence"]
        )

        characters = (
            text_result["characters"]
        )

        # -------------------------------------------------
        # 3. Сохраняем результат
        # -------------------------------------------------

        output_path = (
            output_dir
            / f"{image_path.stem}_text"
            f"{image_path.suffix}"
        )

        cv2.imwrite(
            str(output_path),
            text_crop,
        )

        print(
            f"  bbox: {bbox}"
        )

        print(
            f"  characters: {characters}"
        )

        print(
            f"  confidence: {confidence}"
        )

        print(
            f"  saved: {output_path}"
        )

        print()


if __name__ == "__main__":
    main()