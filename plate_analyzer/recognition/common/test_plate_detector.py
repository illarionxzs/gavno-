from pathlib import Path

import cv2

from recognition.common.plate_detector import PlateDetector
from recognition.common.plate_upscaler import PlateUpscaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGES_DIR = PROJECT_ROOT / "images"
RESULTS_DIR = PROJECT_ROOT / "results" / "plate_detection"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Не найдена папка с изображениями:\n"
            f"{IMAGES_DIR}"
        )

    image_paths = sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        print(
            f"В папке нет изображений:\n"
            f"{IMAGES_DIR}"
        )
        return

    detector = PlateDetector()
    upscaler = PlateUpscaler()

    print()
    print("=== ML PLATE DETECTOR ===")
    print()
    print("Images:", len(image_paths))
    print("Input:", IMAGES_DIR)
    print("Output:", RESULTS_DIR)
    print()

    for image_path in image_paths:

        print("-" * 60)
        print("Image:", image_path.name)

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            print("ERROR: не удалось загрузить изображение")
            continue

        result = detector.detect(image)

        print(
            "bbox:",
            result["bbox"],
        )

        print(
            "confidence:",
            result["confidence"],
        )

        print("detections:")

        for i, detection in enumerate(result["detections"]):
            print(
                i,
                "confidence:",
                detection["confidence"],
                "bbox:",
                detection["bbox"],
                "detection_bbox:",
                detection["detection_bbox"],
            )

        print(
            "detections:",
            len(result["detections"]),
        )

        if result["crop"] is None:
            print("Plate: NOT FOUND")
            continue

        plate = result["crop"]

        print(
            "plate size:",
            plate.shape[1],
            "x",
            plate.shape[0],
        )

        # -------------------------
        # Upscale
        # -------------------------

        upscaled = upscaler.upscale(
            plate
        )

        print(
            "scale:",
            upscaled["scale"],
        )

        # -------------------------
        # Save plate crop
        # -------------------------

        plate_output = (
            RESULTS_DIR
            / f"{image_path.stem}_plate.jpg"
        )

        cv2.imwrite(
            str(plate_output),
            upscaled["image"],
        )

        # -------------------------
        # Save debug image
        # -------------------------

        if result["bbox"] is not None:

            x1, y1, x2, y2 = result["bbox"]

            debug_image = image.copy()

            cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            debug_output = (
                RESULTS_DIR
                / f"{image_path.stem}_debug.jpg"
            )

            cv2.imwrite(
                str(debug_output),
                debug_image,
            )

        print(
            "saved:",
            plate_output.name,
        )

    print()
    print("=" * 60)
    print("Готово.")
    print("Результаты:")
    print(RESULTS_DIR)



if __name__ == "__main__":
    main()