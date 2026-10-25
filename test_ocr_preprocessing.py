
import cv2
import requests
from rapidocr import RapidOCR

URL = (
    "https://raw.githubusercontent.com/"
    "illarionxzs/-/main/10740361_plate_x3.jpg"
)

response = requests.get(URL, timeout=20)
response.raise_for_status()

with open("test_plate.jpg", "wb") as f:
    f.write(response.content)

image = cv2.imread("test_plate.jpg")

if image is None:
    raise RuntimeError("Не удалось открыть изображение")

print("Размер:", image.shape)

ocr = RapidOCR()

# 1. Исходное изображение
variants = {"original": image}

# 2. Увеличение
variants["upscaled"] = cv2.resize(
    image,
    None,
    fx=2,
    fy=2,
    interpolation=cv2.INTER_CUBIC,
)

# 3. Увеличение контраста
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)
contrast = clahe.apply(gray)
variants["contrast"] = cv2.cvtColor(
    contrast,
    cv2.COLOR_GRAY2BGR,
)

# 4. Адаптивная бинаризация
binary = cv2.adaptiveThreshold(
    contrast,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    11,
)
variants["binary"] = cv2.cvtColor(
    binary,
    cv2.COLOR_GRAY2BGR,
)

for name, variant in variants.items():
    result = ocr(variant)

    print(f"\n=== {name} ===")
    print(result)
