import cv2
import numpy as np


class TemporaryPlateDetector:
    """
    Детектор временного белорусского номера.

    Основная структура:

        БУКВЫ -> ЗЕЛЁНАЯ ЗОНА -> 4 ЦИФРЫ

    Для надёжной работы предпочтительно использовать
    symbol-level OCR через word_results.

    Fallback:
        если word_results недоступен, используется detections.
    """

    def __init__(
        self,
        min_green_ratio=0.12,
        min_vertical_overlap=0.30,
        search_padding_ratio=0.35,
    ):
        self.min_green_ratio = min_green_ratio
        self.min_vertical_overlap = min_vertical_overlap
        self.search_padding_ratio = search_padding_ratio

    # ================================================================
    # PUBLIC
    # ================================================================

    def detect(
            self,
            image,
            detections,
            word_results=None,
    ):
        if image is None:
            return None

        # 1. Сначала ищем зелёную область независимо от OCR.
        green_mask = self._build_green_mask(image)

        green_box = self._find_green_cluster(
            roi=green_mask,
            offset_x=0,
            offset_y=0,
        )

        if green_box is None:
            return None

        green_x1 = green_box["x1"]
        green_x2 = green_box["x2"]
        green_center = (green_x1 + green_x2) / 2

        # 2. Получаем отдельные символы OCR.
        if word_results is not None:
            symbols = self._prepare_word_symbols(word_results)
        else:
            symbols = self._prepare_symbols(detections or [])

        if not symbols:
            return None

        # Сначала оставляем символы, центры которых находятся
        # слева или справа от зелёной области.
        left_symbols = [
            s for s in symbols
            if s["center_x"] < green_x1
               and s["text"].isalpha()
        ]

        right_symbols = [
            s for s in symbols
            if s["center_x"] > green_x2
               and s["text"].isdigit()
        ]

        if not left_symbols or not right_symbols:
            return None

        # 3. Сортируем символы слева направо.
        left_symbols.sort(key=lambda s: s["center_x"])
        right_symbols.sort(key=lambda s: s["center_x"])

        # Ищем буквенные группы слева и группы из 4 цифр справа.
        letter_groups = self._find_letter_groups(left_symbols)
        digit_groups = self._find_digit_groups(right_symbols)

        candidates = []

        for letters in letter_groups:
            # Буквы должны располагаться слева от разделителя.
            if letters["center_x"] >= green_x1:
                continue

            for digits in digit_groups:
                # Цифры должны располагаться справа от разделителя.
                if digits["center_x"] <= green_x2:
                    continue

                overlap = self._vertical_overlap(letters, digits)

                if overlap < self.min_vertical_overlap:
                    continue

                # Зелёная область должна находиться по высоте
                # примерно на уровне символов.
                char_top = min(letters["y1"], digits["y1"])
                char_bottom = max(letters["y2"], digits["y2"])

                green_top = green_box["y1"]
                green_bottom = green_box["y2"]

                vertical_intersection = max(
                    0,
                    min(char_bottom, green_bottom)
                    - max(char_top, green_top),
                )

                char_height = max(1, char_bottom - char_top)
                green_vertical_overlap = (
                        vertical_intersection / char_height
                )

                if green_vertical_overlap < 0.30:
                    continue

                # Оценка учитывает зелёную область и расположение
                # букв/цифр относительно неё.
                score = (
                        0.50 * min(1.0, green_box["density"] / 0.50)
                        + 0.30 * overlap
                        + 0.20 * green_vertical_overlap
                )

                candidates.append({
                    "letter_text": letters["text"],
                    "digit_text": digits["text"],
                    "green_x1": green_x1,
                    "green_x2": green_x2,
                    "green_y1": green_box["y1"],
                    "green_y2": green_box["y2"],
                    "green_ratio": green_box["density"],
                    "green_search_x1": 0,
                    "green_search_x2": image.shape[1],
                    "gap_width": digits["x1"] - letters["x2"],
                    "character_height": min(
                        letters["height"],
                        digits["height"],
                    ),
                    "vertical_overlap": overlap,
                    "score": float(score),
                })

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        best = candidates[0]

        return {
            "detected": True,
            **best,
        }

    # ================================================================
    # WORD_RESULTS -> SYMBOLS
    # ================================================================

    def _prepare_word_symbols(self, word_results):
        """
        Преобразует RapidOCR word_results в список отдельных
        OCR-символов.

        Ожидаемый формат:

            [
                [
                    (char, score, box),
                    ...
                ],
                ...
            ]
        """

        symbols = []

        for block in word_results:

            if not isinstance(block, (list, tuple)):
                continue

            for item in block:

                if not isinstance(item, (list, tuple)):
                    continue

                if len(item) != 3:
                    continue

                char, score, box = item

                if char is None:
                    continue

                char = str(char).strip().upper()

                if not char:
                    continue

                if not box or len(box) != 4:
                    continue

                try:
                    points = np.asarray(
                        box,
                        dtype=np.float32,
                    )
                except (TypeError, ValueError):
                    continue

                if points.shape != (4, 2):
                    continue

                x1 = float(
                    np.min(points[:, 0])
                )
                x2 = float(
                    np.max(points[:, 0])
                )
                y1 = float(
                    np.min(points[:, 1])
                )
                y2 = float(
                    np.max(points[:, 1])
                )

                width = x2 - x1
                height = y2 - y1

                if width <= 0 or height <= 0:
                    continue

                try:
                    confidence = float(score)
                except (
                    TypeError,
                    ValueError,
                ):
                    confidence = 0.0

                symbols.append({
                    "text": char,
                    "confidence": confidence,

                    "x1": x1,
                    "x2": x2,
                    "y1": y1,
                    "y2": y2,

                    "width": width,
                    "height": height,

                    "center_x": (
                        x1 + x2
                    ) / 2,

                    "center_y": (
                        y1 + y2
                    ) / 2,
                })

        symbols.sort(
            key=lambda item: (
                item["center_y"],
                item["x1"],
            )
        )

        return symbols

    # ================================================================
    # FALLBACK OCR SYMBOLS
    # ================================================================

    def _prepare_symbols(self, detections):
        """
        Fallback для старого OCRRecognizer.

        Если detection содержит текст из нескольких символов,
        он рассматривается как один OCR-блок.
        """

        symbols = []

        for detection in detections:

            text = str(
                detection.get("text", "")
            ).strip().upper()

            if not text:
                continue

            box = detection.get("box")

            if box is None:
                continue

            try:
                points = np.asarray(
                    box,
                    dtype=np.float32,
                )
            except (TypeError, ValueError):
                continue

            if points.size == 0:
                continue

            x1 = float(
                np.min(points[:, 0])
            )
            x2 = float(
                np.max(points[:, 0])
            )
            y1 = float(
                np.min(points[:, 1])
            )
            y2 = float(
                np.max(points[:, 1])
            )

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            symbols.append({
                "text": text,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "width": width,
                "height": height,
                "center_x": (
                    x1 + x2
                ) / 2,
                "center_y": (
                    y1 + y2
                ) / 2,
            })

        symbols.sort(
            key=lambda item: item["x1"]
        )

        return symbols

    # ================================================================
    # DIGIT GROUPS
    # ================================================================

    def _find_digit_groups(self, symbols):
        groups = []

        for index in range(
            len(symbols)
        ):

            for length in range(1, 5):

                end = index + length

                if end > len(symbols):
                    continue

                group = symbols[
                    index:end
                ]

                text = "".join(
                    item["text"]
                    for item in group
                )

                if len(text) != 4:
                    continue

                if not text.isdigit():
                    continue

                if not self._same_horizontal_line(
                    group
                ):
                    continue

                groups.append(
                    self._merge_symbols(
                        group
                    )
                )

        return self._remove_duplicate_groups(
            groups
        )

    # ================================================================
    # LETTER GROUPS
    # ================================================================

    def _find_letter_groups(self, symbols):
        groups = []

        for index in range(
            len(symbols)
        ):

            for length in range(1, 4):

                end = index + length

                if end > len(symbols):
                    continue

                group = symbols[
                    index:end
                ]

                text = "".join(
                    item["text"]
                    for item in group
                )

                if not text:
                    continue

                if not text.isalpha():
                    continue

                if not self._same_horizontal_line(
                    group
                ):
                    continue

                groups.append(
                    self._merge_symbols(
                        group
                    )
                )

        return self._remove_duplicate_groups(
            groups
        )

    # ================================================================
    # GROUP GEOMETRY
    # ================================================================

    def _same_horizontal_line(
        self,
        symbols,
    ):
        if not symbols:
            return False

        y_centers = [
            item["center_y"]
            for item in symbols
        ]

        heights = [
            item["height"]
            for item in symbols
        ]

        average_height = (
            sum(heights)
            / len(heights)
        )

        if average_height <= 0:
            return False

        max_vertical_difference = (
            average_height * 0.50
        )

        return (
            max(y_centers)
            - min(y_centers)
            <= max_vertical_difference
        )

    def _merge_symbols(self, symbols):
        text = "".join(
            item["text"]
            for item in symbols
        )

        x1 = min(
            item["x1"]
            for item in symbols
        )

        x2 = max(
            item["x2"]
            for item in symbols
        )

        y1 = min(
            item["y1"]
            for item in symbols
        )

        y2 = max(
            item["y2"]
            for item in symbols
        )

        return {
            "text": text,

            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,

            "width": x2 - x1,
            "height": y2 - y1,

            "center_x": (
                x1 + x2
            ) / 2,

            "center_y": (
                y1 + y2
            ) / 2,
        }

    def _remove_duplicate_groups(
        self,
        groups,
    ):
        unique = {}

        for group in groups:

            key = (
                group["text"],
                round(
                    group["x1"],
                    1,
                ),
                round(
                    group["x2"],
                    1,
                ),
                round(
                    group["y1"],
                    1,
                ),
                round(
                    group["y2"],
                    1,
                ),
            )

            unique[key] = group

        return list(
            unique.values()
        )

    # ================================================================
    # GREEN MASK
    # ================================================================

    def _build_green_mask(self, image):
        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        lower_green = np.array(
            [35, 35, 25],
            dtype=np.uint8,
        )

        upper_green = np.array(
            [100, 255, 255],
            dtype=np.uint8,
        )

        mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green,
        )

        kernel = np.ones(
            (3, 3),
            dtype=np.uint8,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return mask

    # ================================================================
    # CANDIDATE
    # ================================================================

    def _check_candidate(
        self,
        image,
        green_mask,
        letters,
        digits,
    ):
        vertical_overlap = (
            self._vertical_overlap(
                letters,
                digits,
            )
        )

        if (
            vertical_overlap
            < self.min_vertical_overlap
        ):
            return None

        character_height = min(
            letters["height"],
            digits["height"],
        )

        if character_height <= 0:
            return None

        gap_width = (
            digits["x1"]
            - letters["x2"]
        )

        if gap_width <= 0:
            return None

        # ------------------------------------------------------------
        # Поиск зелёной зоны.
        #
        # Padding зависит от высоты символа,
        # а не от абсолютного разрешения.
        # ------------------------------------------------------------

        padding = (
            character_height
            * self.search_padding_ratio
        )

        search_x1 = max(
            0,
            int(
                round(
                    letters["x2"]
                    - padding
                )
            ),
        )

        search_x2 = min(
            green_mask.shape[1],
            int(
                round(
                    digits["x1"]
                    + padding
                )
            ),
        )

        search_y1 = max(
            0,
            int(
                round(
                    min(
                        letters["y1"],
                        digits["y1"],
                    )
                    - character_height
                    * 0.25
                )
            ),
        )

        search_y2 = min(
            green_mask.shape[0],
            int(
                round(
                    max(
                        letters["y2"],
                        digits["y2"],
                    )
                    + character_height
                    * 0.25
                )
            ),
        )

        if search_x2 <= search_x1:
            return None

        if search_y2 <= search_y1:
            return None

        roi = green_mask[
            search_y1:search_y2,
            search_x1:search_x2,
        ]

        if roi.size == 0:
            return None

        # ------------------------------------------------------------
        # Смотрим зелёный цвет по горизонтали.
        # ------------------------------------------------------------

        green_box = (
            self._find_green_cluster(
                roi=roi,
                offset_x=search_x1,
                offset_y=search_y1,
            )
        )

        if green_box is None:
            return None

        green_ratio = (
            green_box["green_pixels"]
            / green_box["area"]
            if green_box["area"] > 0
            else 0.0
        )

        # Для общей области между буквами и цифрами
        # используем дополнительную метрику.
        gap_roi_x1 = max(
            0,
            int(
                round(
                    letters["x2"]
                )
            ),
        )

        gap_roi_x2 = min(
            green_mask.shape[1],
            int(
                round(
                    digits["x1"]
                )
            ),
        )

        if gap_roi_x2 <= gap_roi_x1:
            return None

        gap_roi = green_mask[
            search_y1:search_y2,
            gap_roi_x1:gap_roi_x2,
        ]

        if gap_roi.size == 0:
            return None

        gap_green_ratio = (
            cv2.countNonZero(
                gap_roi
            )
            / gap_roi.size
        )

        if (
            gap_green_ratio
            < self.min_green_ratio
        ):
            return None

        score = self._calculate_score(
            green_ratio=green_ratio,
            gap_green_ratio=gap_green_ratio,
            vertical_overlap=vertical_overlap,
            green_box=green_box,
            letters=letters,
            digits=digits,
            character_height=character_height,
        )

        return {
            "letter_text":
                letters["text"],

            "digit_text":
                digits["text"],

            "letter_x1":
                letters["x1"],

            "letter_x2":
                letters["x2"],

            "digit_x1":
                digits["x1"],

            "digit_x2":
                digits["x2"],

            "gap_width":
                gap_width,

            "character_height":
                character_height,

            "green_search_x1":
                search_x1,

            "green_search_x2":
                search_x2,

            "green_x1":
                green_box["x1"],

            "green_x2":
                green_box["x2"],

            "green_y1":
                green_box["y1"],

            "green_y2":
                green_box["y2"],

            "green_ratio":
                gap_green_ratio,

            "vertical_overlap":
                vertical_overlap,

            "score":
                score,
        }

    # ================================================================
    # GREEN PROFILE
    # ================================================================

    def _find_green_cluster(
        self,
        roi,
        offset_x,
        offset_y,
    ):
        """
        Находит не отдельный contour, а горизонтальный
        кластер зелёных столбцов.

        Это важно:

            старый вариант:
                самый плотный contour

            новый вариант:
                значимая непрерывная зелёная зона

        Поэтому маленький шумовой contour больше
        не сможет победить настоящую зелёную полосу.
        """

        if roi is None or roi.size == 0:
            return None

        height, width = roi.shape

        if height <= 0 or width <= 0:
            return None

        # ------------------------------------------------------------
        # Доля зелёного в каждом столбце.
        # ------------------------------------------------------------

        column_percent = np.count_nonzero(
            roi,
            axis=0,
        ) / height

        # Небольшой порог.
        #
        # Не используем абсолютную ширину.
        threshold = 0.08

        active = (
            column_percent
            >= threshold
        )

        # ------------------------------------------------------------
        # Находим непрерывные диапазоны.
        # ------------------------------------------------------------

        ranges = []

        start = None

        for x, is_active in enumerate(
            active
        ):

            if is_active and start is None:
                start = x

            elif (
                not is_active
                and start is not None
            ):
                ranges.append(
                    (
                        start,
                        x - 1,
                    )
                )

                start = None

        if start is not None:
            ranges.append(
                (
                    start,
                    width - 1,
                )
            )

        if not ranges:
            return None

        # ------------------------------------------------------------
        # Объединяем близкие зелёные диапазоны.
        #
        # Маленькие разрывы могут появляться из-за
        # шума / JPEG / бликов.
        # ------------------------------------------------------------

        merged = []

        max_gap = max(
            1,
            int(
                round(
                    width * 0.03
                )
            ),
        )

        for current in ranges:

            if not merged:
                merged.append(
                    list(current)
                )
                continue

            previous = merged[-1]

            distance = (
                current[0]
                - previous[1]
                - 1
            )

            if distance <= max_gap:
                previous[1] = current[1]
            else:
                merged.append(
                    list(current)
                )

        # ------------------------------------------------------------
        # Оцениваем каждый кластер.
        # ------------------------------------------------------------

        candidates = []

        for x1, x2 in merged:

            cluster = roi[
                :,
                x1:x2 + 1,
            ]

            if cluster.size == 0:
                continue

            green_pixels = cv2.countNonZero(
                cluster
            )

            cluster_area = cluster.size

            density = (
                green_pixels
                / cluster_area
            )

            cluster_width = (
                x2 - x1 + 1
            )

            width_ratio = (
                cluster_width
                / width
            )

            # --------------------------------------------------------
            # Реальный зелёный блок должен иметь:
            #
            # - заметную ширину;
            # - достаточно зелёных пикселей;
            # - относительно высокую плотность.
            #
            # Но размеры остаются относительными.
            # --------------------------------------------------------

            if cluster_width < 2:
                continue

            # --------------------------------------------------------
            # Вертикальный диапазон зелени.
            # --------------------------------------------------------

            rows = np.where(
                np.any(
                    cluster > 0,
                    axis=1,
                )
            )[0]

            if len(rows) == 0:
                continue

            y1 = int(rows[0])
            y2 = int(rows[-1])

            actual_height = (
                y2 - y1 + 1
            )

            # --------------------------------------------------------
            # Оценка.
            #
            # Ширина и количество зелёного важнее
            # микроскопического плотного шума.
            # --------------------------------------------------------

            score = (
                density * 0.45
                + min(
                    width_ratio / 0.20,
                    1.0,
                ) * 0.35
                + min(
                    green_pixels
                    / max(
                        1,
                        height * 3,
                    ),
                    1.0,
                ) * 0.20
            )

            candidates.append({
                "x1":
                    x1 + offset_x,

                "x2":
                    x2 + 1 + offset_x,

                "y1":
                    y1 + offset_y,

                "y2":
                    y2 + 1 + offset_y,

                "width":
                    cluster_width,

                "height":
                    actual_height,

                "density":
                    density,

                "width_ratio":
                    width_ratio,

                "green_pixels":
                    green_pixels,

                "area":
                    cluster_area,

                "score":
                    score,
            })

        if not candidates:
            return None

        candidates.sort(
            key=lambda item:
            item["score"],
            reverse=True,
        )

        return candidates[0]

    # ================================================================
    # VERTICAL OVERLAP
    # ================================================================

    def _vertical_overlap(
        self,
        first,
        second,
    ):
        top = max(
            first["y1"],
            second["y1"],
        )

        bottom = min(
            first["y2"],
            second["y2"],
        )

        overlap = (
            bottom - top
        )

        if overlap <= 0:
            return 0.0

        first_height = (
            first["y2"]
            - first["y1"]
        )

        second_height = (
            second["y2"]
            - second["y1"]
        )

        min_height = min(
            first_height,
            second_height,
        )

        if min_height <= 0:
            return 0.0

        return overlap / min_height

    # ================================================================
    # SCORE
    # ================================================================

    def _calculate_score(
            self,
            green_ratio,
            gap_green_ratio,
            vertical_overlap,
            green_box,
            letters,
            digits,
            character_height,
    ):
        green_score = min(
            1.0,
            green_box["density"] / 0.50,
        )

        vertical_score = min(
            1.0,
            vertical_overlap,
        )

        green_center = (
                               green_box["x1"]
                               + green_box["x2"]
                       ) / 2

        gap_center = (
                             letters["x2"]
                             + digits["x1"]
                     ) / 2

        gap_width = (
                digits["x1"]
                - letters["x2"]
        )

        if gap_width <= 0:
            center_score = 0.0
        else:
            center_distance = abs(
                green_center - gap_center
            )
            normalized_distance = (
                    center_distance / gap_width
            )
            center_score = max(
                0.0,
                1.0 - normalized_distance,
            )

        letter_length_score = (
                min(len(letters["text"]), 2) / 2
        )

        digit_length_score = (
            1.0 if len(digits["text"]) == 4 else 0.0
        )

        score = (
                0.35 * green_score
                + 0.25 * vertical_score
                + 0.15 * center_score
                + 0.15 * letter_length_score
                + 0.10 * digit_length_score
        )

        return float(score)
