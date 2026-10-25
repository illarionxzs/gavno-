# import re
#
# COUNTRY_CODES = {
#     "BY": "Беларусь",
#     "RU": "Россия",
#     'MD': 'Молдова'
# }
import re
#сюда добавляются коды стран для того, чтобы они учитывались в статистике, ещё надо будет добавить маски этих номеров в словарь INTERNATIONAL_PATTERNS в main.py
COUNTRY_CODES = {
    "KZ": "Казахстан",
    "MD": "Молдова",
    "AM": "Армения",
    "PL": "Польша",
}

def clean_text(text: str) -> str:
    """Очищает текст от спецсимволов и приводит к верхнему регистру."""
    if not text:
        return ""
    return re.sub(r"[^A-Z0-9]", "", text.upper())

def normalize_detections(detections: list) -> dict | None:
    """Универсальный нормализатор: только очистка и поиск явных кодов."""
    if not detections:
        return None

    # Ищем явный код страны (если он есть на знаке, например, KZ или MD)
    country_code = None
    country = None
    for detection in detections:
        text = clean_text(detection["text"])
        if text in COUNTRY_CODES:
            country_code = text
            country = COUNTRY_CODES[text]
            break

    # Объединяем все куски OCR в единую строку номера
    combined_text = "".join([clean_text(d["text"]) for d in detections if clean_text(d["text"])])

    if not combined_text:
        return None

    # Если нашли явный код страны, убираем его из текста номера
    if country_code and combined_text.startswith(country_code):
        combined_text = combined_text[len(country_code):]

    confidences = [float(d["confidence"]) for d in detections if "confidence" in d]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "raw_detections": detections,
        "number": combined_text,
        "country_code": country_code,  # Будет None для РФ и РБ, и "MD"/"KZ" для остальных
        "country": country,
        "confidence": confidence,
    }