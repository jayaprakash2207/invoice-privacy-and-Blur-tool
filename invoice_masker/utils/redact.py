import random
import string
from typing import Iterable, Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
from faker import Faker


def _fit_text(
    text: str, box_w: int, box_h: int, font_face: int, thickness: int
) -> Tuple[float, Tuple[int, int]]:
    font_scale = 0.9
    while font_scale > 0.3:
        (tw, th), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        if tw <= box_w - 4 and th <= box_h - 4:
            return font_scale, (tw, th)
        font_scale -= 0.1
    return 0.0, (0, 0)


def _dummy_text(original: str, charset: str, min_len: int = 4) -> str:
    length = max(min_len, len(original))
    return "".join(random.choice(charset) for _ in range(length))


def redact_with_dummy_text(
    image: np.ndarray,
    ocr_results: Iterable[Tuple[Iterable[Iterable[int]], str, float]],
    sensitive_boxes: Iterable[Iterable[int]],
    details: Optional[List[Dict[str, Any]]] = None,
    smart_faker: bool = False,
    charset: str = string.ascii_uppercase + string.digits,
    min_len: int = 4,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    output = image.copy()
    box_map = {tuple(box): (text, conf) for box, text, conf in _zip_boxes(ocr_results)}
    detail_map = {tuple(item["bbox"]): item for item in (details or [])}
    faker = Faker() if smart_faker else None

    for box in sensitive_boxes:
        if len(box) != 4:
            continue
        x_min, y_min, x_max, y_max = box
        if x_max <= x_min or y_max <= y_min:
            continue

        # Fill background
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), background_color, -1)

        # Render dummy text centered
        text, _conf = box_map.get(tuple(box), ("REDACTED", 0.0))
        if smart_faker:
            dummy = _smart_dummy(
                faker,
                text,
                detail_map.get(tuple(box)),
            )
        else:
            dummy = _dummy_text(text, charset, min_len=min_len)
        box_w = x_max - x_min
        box_h = y_max - y_min
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_scale, (tw, th) = _fit_text(dummy, box_w, box_h, font_face, thickness)
        if font_scale <= 0:
            continue
        x = x_min + max(2, (box_w - tw) // 2)
        y = y_min + max(2, (box_h + th) // 2)
        cv2.putText(output, dummy, (x, y), font_face, font_scale, text_color, thickness)

    return output


def _zip_boxes(ocr_results):
    for bbox, text, conf in ocr_results:
        xs = [int(point[0]) for point in bbox]
        ys = [int(point[1]) for point in bbox]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        yield [x_min, y_min, x_max, y_max], text, conf


def _smart_dummy(faker: Faker, original: str, detail: Optional[Dict[str, Any]]) -> str:
    if detail:
        regex_hits = set(detail.get("regex_hits", []))
        ner_labels = set(detail.get("ner_labels", []))

        if "EMAIL" in regex_hits:
            return faker.email()
        if "PHONE" in regex_hits:
            return faker.phone_number()
        if "ADDRESS" in regex_hits:
            return faker.street_address()
        if "ZIP" in regex_hits:
            return faker.postcode()
        if "DATE" in regex_hits:
            return str(faker.date())
        if "INVOICE_ID" in regex_hits or "INVOICE_REF" in regex_hits:
            return f"INV-{faker.random_int(min=10000, max=99999)}"

        if "PERSON" in ner_labels:
            return faker.name()
        if "ORG" in ner_labels:
            return faker.company()
        if "GPE" in ner_labels:
            return faker.city()

        if "KEYWORD" in regex_hits:
            return "REDACTED"

    return _dummy_text(original, string.ascii_uppercase + string.digits, min_len=4)
