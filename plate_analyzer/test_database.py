
from storage.database import PlateDatabase


db = PlateDatabase("data/test_plates.db")

result = {
    "number": "1234AB",
    "country": "Беларусь",
    "country_code": "BY",
    "type": "passenger",
    "region_code": "4",
    "region": "Гродненская область",
    "ocr_corrected": False,
    "ocr_original": None,
}

print(db.add_vehicle(result, "images/photo1.jpg"))
print(db.add_vehicle(result, "images/photo2.jpg"))
print(db.add_vehicle(result, "images/photo1.jpg"))

print(db.get_statistics())
