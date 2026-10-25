
import re

from .temporary_detector import TemporaryPlateDetector


BELARUS_LETTERS = "ABCEHIKMOPTX"

BELARUS_REGIONS = {
    "0": "Военные / специальные ведомства",
    "1": "Брестская область",
    "2": "Витебская область",
    "3": "Гомельская область",
    "4": "Гродненская область",
    "5": "Минская область",
    "6": "Могилёвская область",
    "7": "г. Минск",
    "8": "г. Минск",
}


TYPE_PASSENGER = "passenger"
TYPE_TRUCK = "truck"
TYPE_BUS = "bus"
TYPE_MOTORCYCLE = "motorcycle"
TYPE_TRAILER = "trailer"
TYPE_TAXI = "taxi"
TYPE_TRANSIT = "transit"
TYPE_TEMPORARY = "temporary"
TYPE_TRACTOR = "tractor"
TYPE_DIPLOMATIC = "diplomatic"
TYPE_CONSULAR = "consular"
TYPE_DIPLOMATIC_TRAILER = "diplomatic_trailer"
TYPE_POLICE = "police"
TYPE_MILITARY_SPECIAL = "military_special"
TYPE_ELECTRIC = "electric"
TYPE_OLD = "old"
TYPE_UNKNOWN = "unknown"


POLICE_REGION_CODES = {
    "BN": "1",  # Брестская область
    "BT": "2",  # Витебская область
    "GS": "3",  # Гомельская область
    "GK": "4",  # Гродненская область
    "MO": "5",  # Минская область
    "MG": "6",  # Могилёвская область
    "MH": "7",  # г. Минск
    "MI": "7",  # г. Минск
}


CYRILLIC_TO_LATIN = {
    "А": "A",
    "В": "B",
    "С": "C",
    "Е": "E",
    "Н": "H",
    "І": "I",
    "К": "K",
    "М": "M",
    "О": "O",
    "Р": "P",
    "Т": "T",
    "Х": "X",
    "Б": "B",
    "Г": "G",
}


class BelarusClassifier:

    def __init__(self):
        self.temporary_plate_detector = TemporaryPlateDetector()

    # =========================================================
    # MAIN CLASSIFICATION
    # =========================================================

    def classify(
            self,
            number: str,
            country_code="BY",
            temporary_detected=False,
            plate_color=None,
            vehicle_type=None,
            image=None,
            detections=None,
            word_results=None,
            _allow_ocr_extraction=True,
    ):
        original_ocr = number
        debug_plate_color = plate_color

        print(
            "\n[CLASSIFY START]"
            f" number={number!r}"
            f" plate_color={plate_color!r}"
        )

        # =====================================================
        # 0. ВОЕННЫЕ НОМЕРА ПО СЫРЫМ DETECTIONS (Пока дефис -0 еще жив!)
        # Проверяем в первую очередь, до любой очистки и нормализации.
        # =====================================================
        if detections:
            for det in detections:
                det_text = str(det.get("text", "")).upper()
                det_norm = self._normalize_ocr_letters(det_text)
                if re.search(rf"[{BELARUS_LETTERS}]{{2}}\d{{4}}[\- ]0", det_norm):
                    cleaned_military = self._clean_number(det_text)
                    return self._build_result(
                        number=cleaned_military if cleaned_military else number,
                        vehicle_type=TYPE_MILITARY_SPECIAL,
                        region_code="0",
                    )

        if not number:
            return self._unknown(number)

        # =====================================================
        # 1. ВРЕМЕННЫЕ НОМЕРА (Доверяем строго флагу из main.py)
        # =====================================================
        if temporary_detected:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_TEMPORARY,
                region_code=None,
            )

        # =====================================================
        # CLEAN + NORMALIZE ДЛЯ ОСТАЛЬНЫХ НОМЕРОВ
        # =====================================================
        number = self._clean_number(number)
        if not number:
            return self._unknown(number)

        number = self._normalize_ocr_letters(number)
        if not number:
            return self._unknown(number)

        normalized_plate_color = (
            str(debug_plate_color).strip().lower()
            if debug_plate_color is not None
            else "unknown"
        )

        # =====================================================
        # 2. СПЕЦФОРМАТЫ ВЫСШЕГО ПРИОРИТЕТА (Полиция, Дипломаты, Такси)
        # =====================================================

        # --- ПОЛИЦИЯ ---
        match = re.fullmatch(r"(\d{4})([A-ZА-ЯІ]{2})", number)
        if match:
            police_code = self._normalize_police_code(match.group(2))
            region_code = POLICE_REGION_CODES.get(police_code)
            if region_code is not None:
                return self._build_result(
                    number=number,
                    vehicle_type=TYPE_POLICE,
                    region_code=region_code,
                )

        # --- ДИПЛОМАТЫ ---
        match = re.fullmatch(r"CD(\d{4})([1-8])", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_DIPLOMATIC,
                region_code=match.group(2),
            )

        # --- КОНСУЛЫ ---
        match = re.fullmatch(r"CC(\d{4})([1-8])", number)
        if match and (normalized_plate_color == "red" or country_code != "BY"):
            return self._build_result(
                number=number,
                vehicle_type=TYPE_CONSULAR,
                region_code=match.group(2),
            )

        # --- ТАКСИ ---
        match = re.fullmatch(r"([1-8])(TAX|TBX|TEX)(\d{4})", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_TAXI,
                region_code=match.group(1),
            )

        # =====================================================
        # 3. ТРАНЗИТЫ И ЭЛЕКТРОМОБИЛИ
        # =====================================================
        match = re.fullmatch(rf"([1-8])([{BELARUS_LETTERS}]{{3}})(\d{{4}})", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_TRANSIT,
                region_code=match.group(1),
            )

        match = re.fullmatch(rf"E(\d{{3}})([{BELARUS_LETTERS}]{{2}})([1-8])", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_ELECTRIC,
                region_code=match.group(3),
            )

        match = re.fullmatch(rf"([{BELARUS_LETTERS}]{{2}})E(\d{{3}})([1-8])", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_ELECTRIC,
                region_code=match.group(3),
            )

        match = re.fullmatch(rf"([{BELARUS_LETTERS}]){{2}}BP\d{{4}}", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_TEMPORARY,
                region_code=None,
            )

        # =====================================================
        # 4. ОСНОВНЫЕ ГРАЖДАНСКИЕ ФОРМАТЫ
        # =====================================================

        # Пассажирский стандарт: 1234AB1
        match = re.fullmatch(rf"(\d{{4}})([{BELARUS_LETTERS}]{{2}})([1-8])", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_PASSENGER,
                region_code=match.group(3),
            )

        # Грузовой / автобус: AB12341
        match = re.fullmatch(rf"([{BELARUS_LETTERS}]{{2}})(\d{{4}})([1-8])", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_TRUCK,
                region_code=match.group(3),
            )

        # Квадратные пассажирские номера: AB17630 (2 буквы + регион 1-8 + 4 цифры)
        match = re.fullmatch(rf"([{BELARUS_LETTERS}]{{2}})([1-8])(\d{{4}})", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_PASSENGER,
                region_code=match.group(2),
            )

        # Прицеп: A1234B1
        match = re.fullmatch(
            rf"([{BELARUS_LETTERS}])(\d{{4}})([{BELARUS_LETTERS}])([1-8])",
            number,
        )
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_TRAILER,
                region_code=match.group(4),
            )

        # Дипломатический прицеп: 1234A1
        match = re.fullmatch(rf"(\d{{4}})([{BELARUS_LETTERS}])([1-8])", number)
        if match:
            return self._build_result(
                number=number,
                vehicle_type=TYPE_DIPLOMATIC_TRAILER,
                region_code=match.group(3),
            )

        # =====================================================
        # 5. СТАРЫЕ ФОРМАТЫ
        # =====================================================
        old_patterns = [
            rf"\d{{4}}[{BELARUS_LETTERS}]{{2}}",
            rf"\d{{4}}[{BELARUS_LETTERS}]{{3}}",
            rf"[{BELARUS_LETTERS}]{{2}}\d{{4}}",
            rf"[{BELARUS_LETTERS}]\d{{4}}[{BELARUS_LETTERS}]",
            rf"[{BELARUS_LETTERS}]\d{{5}}",
        ]
        for pattern in old_patterns:
            if re.fullmatch(pattern, number):
                return self._build_result(
                    number=number,
                    vehicle_type=TYPE_OLD,
                    region_code=None,
                )

        # =====================================================
        # 6. ФИНАЛЬНЫЙ ПОИСК ВНУТРИ OCR И UNKNOWN
        # =====================================================
        if _allow_ocr_extraction:
            candidates = self._extract_number_candidates_from_ocr(
                original_ocr,
                word_results=word_results,
                plate_color=plate_color,
            )
            if candidates:
                best_candidate = candidates[0]
                result = best_candidate["result"]
                result["ocr_corrected"] = True
                result["ocr_original"] = original_ocr
                return result

        return self._unknown(number)

    # =========================================================
    # OCR CANDIDATE EXTRACTION
    # =========================================================

    def _extract_number_candidates_from_ocr(
        self,
        ocr_text,
        word_results=None,
        plate_color=None
    ):
        """
        Ищет настоящий номер среди OCR.

        Приоритет:
        1. отдельный OCR-блок;
        2. номер внутри OCR-блока;
        3. группа OCR-блоков;
        4. отдельные токены полного OCR;
        5. номер внутри полного OCR.

        Текст рамки не фильтруется по blacklist.
        """

        if not ocr_text and not word_results:
            return []

        candidates = []

        def add_candidate(
            value,
            source,
            geometry=None,
        ):
            if not value:
                return

            value = str(value).upper().strip()

            normalized = self._normalize_candidate_ocr(
                value
            )

            if not normalized:
                return

            if len(normalized) < 5:
                return

            if len(normalized) > 10:
                return

            corrected_variants = (
                self._generate_ocr_variants(
                    normalized
                )
            )

            for corrected in corrected_variants:

                if not corrected:
                    continue

                if len(corrected) < 5:
                    continue

                if len(corrected) > 10:
                    continue

                duplicate = False

                for existing in candidates:
                    if (
                        existing["number"] == corrected
                        and self._same_geometry(
                            existing.get("geometry"),
                            geometry,
                        )
                    ):
                        duplicate = True
                        break

                if duplicate:
                    continue

                result = self.classify(
                    number=corrected,
                    country_code="BY",
                    temporary_detected=False,
                    plate_color=plate_color,
                    vehicle_type=None,
                    image=None,
                    detections=None,
                    word_results=None,
                    _allow_ocr_extraction=False,
                )

                if result.get("type") == TYPE_UNKNOWN:
                    continue

                candidates.append({
                    "number": corrected,
                    "result": result,
                    "source": source,
                    "geometry": geometry,
                })

        # =====================================================
        # 1. INDIVIDUAL OCR BLOCKS
        # =====================================================

        blocks = self._normalize_ocr_blocks(
            word_results
        )

        for block in blocks:

            text = block["text"]

            # Весь блок может сам являться номером.
            add_candidate(
                text,
                source="ocr_block",
                geometry=block,
            )

            # Номер может находиться внутри блока:
            #
            # 6666 1K-7
            #
            compact = re.sub(
                r"[^A-Z0-9]",
                "",
                self._normalize_ocr_letters(
                    text.upper()
                ),
            )

            for candidate in (
                self._extract_embedded_candidates(
                    compact
                )
            ):
                add_candidate(
                    candidate,
                    source="ocr_block_embedded",
                    geometry=block,
                )

        # =====================================================
        # 2. GROUPS OF OCR BLOCKS
        # =====================================================

        if len(blocks) >= 2:

            sorted_blocks = sorted(
                blocks,
                key=lambda item: item["center_x"],
            )

            for start in range(
                len(sorted_blocks)
            ):

                combined = ""
                group = []

                for end in range(
                    start,
                    min(
                        start + 3,
                        len(sorted_blocks),
                    ),
                ):

                    block = sorted_blocks[end]

                    combined += " " + block["text"]
                    group.append(block)

                    geometry = (
                        self._merge_ocr_geometry(
                            group
                        )
                    )

                    compact = re.sub(
                        r"[^A-Z0-9]",
                        "",
                        self._normalize_ocr_letters(
                            combined.upper()
                        ),
                    )

                    if compact:

                        for candidate in (
                            self._extract_embedded_candidates(
                                compact
                            )
                        ):
                            add_candidate(
                                candidate,
                                source=(
                                    f"ocr_blocks_"
                                    f"{len(group)}"
                                ),
                                geometry=geometry,
                            )

        # =====================================================
        # 3. FULL OCR TOKENS
        # =====================================================

        original_text = str(
            ocr_text or ""
        ).upper()

        text = self._normalize_ocr_letters(
            original_text
        )

        text = re.sub(
            r"\bBELARUS\b",
            " ",
            text,
        )

        text = re.sub(
            r"[^A-Z0-9]+",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        ).strip()

        if text:

            tokens = text.split()

            for token in tokens:
                add_candidate(
                    token,
                    source="token",
                    geometry=None,
                )

            for index in range(
                len(tokens)
            ):

                for size in (2, 3):

                    if index + size > len(tokens):
                        continue

                    combined = "".join(
                        tokens[
                            index:index + size
                        ]
                    )

                    add_candidate(
                        combined,
                        source=f"tokens_{size}",
                        geometry=None,
                    )

        # =====================================================
        # 4. FULL COMPACT OCR
        # =====================================================

        compact_text = re.sub(
            r"[^A-Z0-9]",
            "",
            original_text,
        )

        if compact_text:

            for candidate in (
                self._extract_embedded_candidates(
                    compact_text
                )
            ):
                add_candidate(
                    candidate,
                    source="embedded",
                    geometry=None,
                )

        if not candidates:
            return []

        # =====================================================
        # 5. SCORING
        # =====================================================

        scored = []

        for item in candidates:

            candidate = item["number"]
            result = item["result"]
            geometry = item.get("geometry")

            score = 0
            source = item["source"]

            if source == "ocr_block":
                score += 45

            elif source == "ocr_block_embedded":
                score += 40

            elif source.startswith("ocr_blocks_"):
                score += 30

            elif source == "token":
                score += 25

            elif source == "tokens_2":
                score += 20

            elif source == "tokens_3":
                score += 15

            elif source == "embedded":
                score += 10

            # Кандидат уже прошёл format validation.
            score += 30

            # Современный белорусский номер обычно 7 символов.
            if len(candidate) == 7:
                score += 15

            if any(
                char.isalpha()
                for char in candidate
            ):
                score += 5

            if any(
                char.isdigit()
                for char in candidate
            ):
                score += 5

            if geometry:
                score += self._score_ocr_geometry(
                    geometry
                )

            # Старый формат должен проигрывать
            # нормальному современному номеру.
            if result.get("type") == TYPE_OLD:
                score -= 15

            scored.append({
                "number": candidate,
                "result": result,
                "score": score,
                "source": source,
                "geometry": geometry,
            })

        scored.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored

    # =========================================================
    # OCR NORMALIZATION
    # =========================================================

    def _normalize_candidate_ocr(
        self,
        value,
    ):
        """
        Нормализует OCR-кандидат.

        ВАЖНО:
        здесь нет глобальной замены 1 -> I.
        """

        value = self._normalize_ocr_letters(
            value.upper()
        )

        value = re.sub(
            r"[^A-Z0-9]",
            "",
            value,
        )

        return value

    def _generate_ocr_variants(
        self,
        value,
    ):
        """
        Генерирует ограниченные варианты OCR.

        Исправления выполняются только в тех позициях,
        где формат номера ожидает букву или цифру.
        """

        variants = {value}

        def add_variant(chars):
            variants.add(
                "".join(chars)
            )

        if len(value) == 7:

            chars = list(value)

            # -------------------------------------------------
            # PASSENGER
            #
            # 1234AB1
            #
            # Позиции 4 и 5 — буквы.
            # -------------------------------------------------

            for position in (4, 5):

                char = chars[position]

                if char == "1":

                    corrected = chars.copy()
                    corrected[position] = "I"

                    add_variant(corrected)

                elif char == "0":

                    corrected = chars.copy()
                    corrected[position] = "O"

                    add_variant(corrected)

            # Последняя позиция — цифра.
            if chars[6] == "I":

                corrected = chars.copy()
                corrected[6] = "1"

                add_variant(corrected)

            # -------------------------------------------------
            # TRUCK / BUS
            #
            # AB12341
            #
            # Первые две позиции — буквы.
            # -------------------------------------------------

            for position in (0, 1):

                char = chars[position]

                if char == "1":

                    corrected = chars.copy()
                    corrected[position] = "I"

                    add_variant(corrected)

                elif char == "0":

                    corrected = chars.copy()
                    corrected[position] = "O"

                    add_variant(corrected)

            # -------------------------------------------------
            # TRAILER
            #
            # A1234B1
            #
            # Позиции 0 и 5 — буквы.
            # -------------------------------------------------

            for position in (0, 5):

                char = chars[position]

                if char == "1":

                    corrected = chars.copy()
                    corrected[position] = "I"

                    add_variant(corrected)

                elif char == "0":

                    corrected = chars.copy()
                    corrected[position] = "O"

                    add_variant(corrected)

        # -------------------------------------------------
        # POLICE
        #
        # 1234XX
        #
        # Последние две позиции — буквенный код.
        #
        # OCR иногда распознаёт G как F.
        # -------------------------------------------------

        if len(value) == 6:

            chars = list(value)

            # OCR: G -> F
            # Возможная обратная коррекция F -> G
            # добавляется только как вариант.
            for position in (4, 5):

                if chars[position] == "F":

                    corrected = chars.copy()
                    corrected[position] = "G"

                    add_variant(corrected)


        id = "u3j9sq"
        # -------------------------------------------------
        # POLICE OCR RECOVERY
        #
        # Нормальный формат:
        #
        # 1234XX
        #
        # OCR иногда цепляет лишний символ слева:
        #
        # 0846MH -> 30846MH
        # 0442GK -> 10442TK
        #
        # Сначала убираем лишний первый символ.
        # -------------------------------------------------


        # -------------------------------------------------
        # POLICE OCR RECOVERY
        #
        # 10442TK -> 0442TK -> 0442GK
        # -------------------------------------------------

        if len(value) == 7:

            without_first = value[1:]

            if (
                    without_first[:4].isdigit()
                    and without_first[4:].isalpha()
            ):
                # Убираем лишний символ слева.
                add_variant(list(without_first))

                # Пробуем исправить T на G в буквенном коде.
                police_code = without_first[4:]

                for position in range(2):

                    if police_code[position] != "T":
                        continue

                    corrected_code = (
                            police_code[:position]
                            + "G"
                            + police_code[position + 1:]
                    )

                    if corrected_code in POLICE_REGION_CODES:
                        corrected = list(without_first)
                        corrected[4:] = corrected_code
                        add_variant(corrected)




        # -------------------------------------------------
        # OLD PLATE OCR RECOVERY
        # 2222M0 -> 2222MO
        # -------------------------------------------------
        if len(value) == 6 and value[:4].isdigit():
            chars = list(value)

            for position in (4, 5):
                if chars[position] == "0":
                    corrected = chars.copy()
                    corrected[position] = "O"
                    add_variant(corrected)



        return list(variants)

    # =========================================================
    # OCR BLOCKS
    # =========================================================

    def _normalize_ocr_blocks(
        self,
        word_results,
    ):
        """
        Приводит OCR blocks к единому формату.
        """

        if not word_results:
            return []

        blocks = []

        for item in word_results:

            text = None
            confidence = 0.0
            box = None

            # -------------------------------------------------
            # dict
            # -------------------------------------------------

            if isinstance(item, dict):

                text = (
                    item.get("text")
                    or item.get("label")
                    or item.get("value")
                )

                confidence = float(
                    item.get(
                        "confidence",
                        item.get(
                            "conf",
                            0.0,
                        ),
                    )
                    or 0.0
                )

                box = (
                    item.get("box")
                    or item.get("bbox")
                    or item.get("points")
                )

            # -------------------------------------------------
            # tuple / list
            # -------------------------------------------------

            elif isinstance(
                item,
                (list, tuple),
            ):

                if len(item) >= 3:

                    # (text, confidence, box)
                    if isinstance(
                        item[0],
                        str,
                    ):

                        text = item[0]

                        confidence = float(
                            item[1] or 0.0
                        )

                        box = item[2]

                    # (box, text, confidence)
                    elif isinstance(
                        item[1],
                        str,
                    ):

                        box = item[0]

                        text = item[1]

                        confidence = float(
                            item[2] or 0.0
                        )

            if not text or not box:
                continue

            coordinates = self._box_to_xyxy(
                box
            )

            if coordinates is None:
                continue

            x1, y1, x2, y2 = coordinates

            width = max(
                0.0,
                x2 - x1,
            )

            height = max(
                0.0,
                y2 - y1,
            )

            if width <= 0 or height <= 0:
                continue

            blocks.append({
                "text": str(text).strip(),
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": height,
                "center_x": (
                    x1 + x2
                ) / 2.0,
                "center_y": (
                    y1 + y2
                ) / 2.0,
            })

        return blocks

    def _box_to_xyxy(
        self,
        box,
    ):
        """
        Поддерживает:

        [x1, y1, x2, y2]

        и:

        [[x1,y1], [x2,y2], ...]
        """

        try:

            if len(box) != 4:
                return None

            # [x1, y1, x2, y2]
            if all(
                isinstance(
                    value,
                    (int, float),
                )
                for value in box
            ):

                return (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )

            # Polygon / quadrilateral
            points = [
                (
                    float(point[0]),
                    float(point[1]),
                )
                for point in box
            ]

            xs = [
                point[0]
                for point in points
            ]

            ys = [
                point[1]
                for point in points
            ]

            return (
                min(xs),
                min(ys),
                max(xs),
                max(ys),
            )

        except (
            TypeError,
            ValueError,
            IndexError,
        ):
            return None

    # =========================================================
    # OCR GEOMETRY
    # =========================================================

    def _score_ocr_geometry(
        self,
        geometry,
    ):
        """
        Оценивает OCR-блок как возможную область номера.

        Используется только геометрия.
        Никакого blacklist текста рамки.
        """

        if not geometry:
            return 0

        score = 0

        width = geometry.get(
            "width",
            0.0,
        )

        height = geometry.get(
            "height",
            0.0,
        )

        confidence = geometry.get(
            "confidence",
            0.0,
        )

        if width <= 0 or height <= 0:
            return 0

        aspect_ratio = (
            width / height
        )

        # Номер обычно существенно шире высоты.
        if aspect_ratio >= 2.5:
            score += 15

        elif aspect_ratio >= 1.8:
            score += 10

        elif aspect_ratio >= 1.3:
            score += 5

        # Высота OCR-блока.
        if height >= 40:
            score += 10

        elif height >= 25:
            score += 5

        # OCR confidence.
        if confidence >= 0.90:
            score += 8

        elif confidence >= 0.75:
            score += 5

        elif confidence >= 0.50:
            score += 2

        return score

    def _merge_ocr_geometry(
        self,
        blocks,
    ):
        if not blocks:
            return None

        x1 = min(
            block["x1"]
            for block in blocks
        )

        y1 = min(
            block["y1"]
            for block in blocks
        )

        x2 = max(
            block["x2"]
            for block in blocks
        )

        y2 = max(
            block["y2"]
            for block in blocks
        )

        confidence = (
            sum(
                block["confidence"]
                for block in blocks
            )
            / len(blocks)
        )

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        return {
            "text": " ".join(
                block["text"]
                for block in blocks
            ),
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": width,
            "height": height,
            "center_x": (
                x1 + x2
            ) / 2.0,
            "center_y": (
                y1 + y2
            ) / 2.0,
        }

    def _same_geometry(
        self,
        first,
        second,
    ):
        if first is None or second is None:
            return first is second

        return (
            abs(
                first.get("x1", 0)
                - second.get("x1", 0)
            ) < 1
            and
            abs(
                first.get("y1", 0)
                - second.get("y1", 0)
            ) < 1
            and
            abs(
                first.get("x2", 0)
                - second.get("x2", 0)
            ) < 1
            and
            abs(
                first.get("y2", 0)
                - second.get("y2", 0)
            ) < 1
        )

    # =========================================================
    # EMBEDDED OCR CANDIDATES
    # =========================================================

    def _extract_embedded_candidates(
        self,
        compact_text,
    ):
        """
        Ищет потенциальные номера внутри OCR-строки.

        Например:

            66661K7AUTOSERVICE

        может дать:

            66661K7
        """

        if not compact_text:
            return []

        patterns = [
            # Passenger
            r"\d{4}[ABCEHIKMOPTX0-9]{2}\d",

            # Truck / bus
            r"[ABCEHIKMOPTX0-9]{2}\d{5}",

            # Trailer
            r"[ABCEHIKMOPTX0-9]\d{4}[ABCEHIKMOPTX0-9]\d",

            # Taxi
            r"1(?:TAX|TBX|TEX)\d{4}",

            # Diplomatic
            r"CD\d{4}[1-8]",
            r"CC\d{4}[1-8]",

            # Electric
            r"E\d{3}[ABCEHIKMOPTX0-9]{2}[1-8]",
            r"[ABCEHIKMOPTX0-9]{2}E\d{3}[1-8]",

            # Temporary
            r"[ABCEHIKMOPTX0-9]{2}BP\d{4}",

            # Transit
            r"[1-8][ABCEHIKMOPTX0-9]{3}\d{4}",

            # Military
            r"[ABCEHIKMOPTX0-9]{2}\d{4}0",

            # Old / partial formats
            r"[ABCEHIKMOPTX0-9]{2}\d{4}",
            r"\d{4}[ABCEHIKMOPTX0-9]{2}",
        ]

        result = []

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                compact_text,
            ):

                value = match.group(0)

                if value not in result:
                    result.append(value)

        return result

    # =========================================================
    # POLICE
    # =========================================================

    def _classify_police(self, number: str):
        match = re.fullmatch(
            r"(\d{4})([A-ZА-ЯІ]{2})",
            number,
        )

        if not match:
            return None

        police_code = self._normalize_police_code(
            match.group(2)
        )

        # Не принимаем неизвестный буквенный код.
        region_code = POLICE_REGION_CODES.get(police_code)

        if region_code is None:
            return None

        return self._build_result(
            number=number,
            vehicle_type=TYPE_POLICE,
            region_code=region_code,
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _clean_number(
        number: str,
    ) -> str:

        if not number:
            return ""

        number = str(
            number
        ).upper()

        number = re.sub(
            r"BELARUS$",
            "",
            number,
        )

        return re.sub(
            r"[^A-ZА-ЯІ0-9]",
            "",
            number,
        )

    @staticmethod
    def _normalize_ocr_letters(
        number: str,
    ) -> str:

        if not number:
            return ""

        result = []

        for char in number:

            result.append(
                CYRILLIC_TO_LATIN.get(
                    char,
                    char,
                )
            )

        return "".join(result)

    @staticmethod
    def _normalize_police_code(
            code: str,
    ) -> str:
        if not code:
            return ""

        code = "".join(
            CYRILLIC_TO_LATIN.get(
                char,
                char,
            )
            for char in code.upper()
        )

        police_fixes = {
            "TK": "GK", "FK": "GK", "EK": "GK", "RK": "GK",
            "TS": "GS", "ES": "GS", "FS": "GS",
            "ME": "MG", "MR": "MG",
        }
        return police_fixes.get(code, code)

    @staticmethod
    def _has_special_region_code(number: str) -> bool:
        if not number:
            return False

        if "TAX" in number or "TBX" in number or "TEX" in number:
            return False

        if re.fullmatch(r"\d{4}[A-ZА-ЯІ0-9]{2}", number):
            return False

        return bool(re.fullmatch(r"[A-Z]{2}\d{4}[ \-]?0", number))

    @staticmethod
    def _build_result(
            number,
            vehicle_type,
            region_code=None,
            ocr_corrected=False,
            ocr_original=None,
    ):
        # Маппинг с учетом OCR-ошибок (TK, EK -> ГК; ME -> МГ)
        police_cyrillic_map = {
            "GK": "ГК", "TK": "ГК", "FK": "ГК", "EK": "ГК", "RK": "ГК",
            "GS": "ГС", "TS": "GS", "ES": "GS", "FS": "GS",
            "MG": "МГ", "ME": "МГ", "MR": "МГ",
            "MH": "МН", "MI": "МИ", "MO": "МО", "BT": "БТ", "BN": "БН"
        }
        if vehicle_type == TYPE_POLICE:
            for lat, cyr in police_cyrillic_map.items():
                if lat in number:
                    number = number.replace(lat, cyr)

        return {
            "number": number,
            "country": "Беларусь",
            "country_code": "BY",
            "type": vehicle_type,
            "region_code": region_code,
            "region": (
                BELARUS_REGIONS.get(
                    region_code
                )
                if region_code
                else None
            ),
            "ocr_corrected": ocr_corrected,
            "ocr_original": ocr_original,
        }

    @staticmethod
    def _unknown(
        number,
    ):

        return {
            "number": number,
            "country": "Беларусь",
            "country_code": "BY",
            "type": TYPE_UNKNOWN,
            "region_code": None,
            "region": None,
            "ocr_corrected": False,
            "ocr_original": None,
        }
