from recognition.belarus.classifier import BelarusClassifier


classifier = BelarusClassifier()


tests = [
    ("1234MH", None),
    ("1234МН", None),
    ("1234MH", "red"),
    ("1234МН", "red"),
]


for number, plate_color in tests:

    result = classifier.classify(
        number=number,
        plate_color=plate_color,
    )

    print(
        f"{number=}, "
        f"{plate_color=}"
    )

    print(
        f"  type: {result['type']}"
    )

    print(
        f"  number: {result['number']}"
    )

    print()