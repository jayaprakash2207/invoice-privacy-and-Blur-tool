import json
import os
import secrets
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Iterable, List

import cv2
import numpy as np


@dataclass
class TokenizerConfig:
    output_dir: str
    db_path: str
    prefix_map: Dict[str, str]


def load_token_db(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_token_db(path: str, db: Dict[str, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def _token(prefix: str) -> str:
    return f"{prefix}{secrets.token_hex(3).upper()}"


def _pick_prefix(detail: Dict[str, Any], prefix_map: Dict[str, str]) -> str:
    regex_hits = set(detail.get("regex_hits", []))
    ner_labels = set(detail.get("ner_labels", []))

    if "EMAIL" in regex_hits:
        return prefix_map.get("EMAIL", "EMAIL_")
    if "PHONE" in regex_hits:
        return prefix_map.get("PHONE", "PHONE_")
    if "ADDRESS" in regex_hits:
        return prefix_map.get("ADDRESS", "ADDR_")
    if "ZIP" in regex_hits:
        return prefix_map.get("ZIP", "ZIP_")
    if "DATE" in regex_hits:
        return prefix_map.get("DATE", "DATE_")
    if "INVOICE_ID" in regex_hits or "INVOICE_REF" in regex_hits:
        return prefix_map.get("INVOICE", "INV_")

    if "PERSON" in ner_labels:
        return prefix_map.get("PERSON", "USER_")
    if "ORG" in ner_labels:
        return prefix_map.get("ORG", "ORG_")
    if "GPE" in ner_labels:
        return prefix_map.get("GPE", "LOC_")

    return prefix_map.get("GEN", "GEN_")


def _fit_text(text: str, box_w: int, box_h: int, font_face: int, thickness: int):
    font_scale = 0.9
    while font_scale > 0.3:
        (tw, th), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        if tw <= box_w - 4 and th <= box_h - 4:
            return font_scale, (tw, th)
        font_scale -= 0.1
    return 0.0, (0, 0)


def tokenize_image(
    image: np.ndarray,
    details: list[dict],
    cfg: TokenizerConfig,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    if image is None or image.size == 0:
        return image, {}

    db = load_token_db(cfg.db_path)
    output = image.copy()
    entries: List[Dict[str, Any]] = []

    for item in details:
        if not item.get("sensitive"):
            continue
        text = item.get("text", "")
        bbox = item.get("bbox", [])
        if not text or len(bbox) != 4:
            continue

        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            continue

        if text not in db:
            prefix = _pick_prefix(item, cfg.prefix_map)
            db[text] = {
                "token": _token(prefix),
                "labels": item.get("ner_labels", []),
                "regex_hits": item.get("regex_hits", []),
            }

        token = db[text]["token"]
        entries.append(
            {
                "token": token,
                "original": text,
                "bbox": [x_min, y_min, x_max, y_max],
            }
        )

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), background_color, -1)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        box_w = x_max - x_min
        box_h = y_max - y_min
        font_scale, (tw, th) = _fit_text(token, box_w, box_h, font_face, thickness)
        if font_scale <= 0:
            continue
        x = x_min + max(2, (box_w - tw) // 2)
        y = y_min + max(2, (box_h + th) // 2)
        cv2.putText(output, token, (x, y), font_face, font_scale, text_color, thickness)

    save_token_db(cfg.db_path, db)
    return output, db, entries


def restore_image_from_tokens(
    image: np.ndarray,
    details: list[dict],
    cfg: TokenizerConfig,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    db = load_token_db(cfg.db_path)
    token_to_text = {value["token"]: key for key, value in db.items()}

    output = image.copy()
    for item in details:
        text = item.get("text", "")
        bbox = item.get("bbox", [])
        if not text or len(bbox) != 4:
            continue

        token = text.strip()
        original = token_to_text.get(token)
        if not original:
            continue

        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            continue

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), background_color, -1)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        box_w = x_max - x_min
        box_h = y_max - y_min
        font_scale, (tw, th) = _fit_text(original, box_w, box_h, font_face, thickness)
        if font_scale <= 0:
            continue
        x = x_min + max(2, (box_w - tw) // 2)
        y = y_min + max(2, (box_h + th) // 2)
        cv2.putText(output, original, (x, y), font_face, font_scale, text_color, thickness)

    return output


def restore_image_from_tokens_ocr(
    image: np.ndarray,
    ocr_results: Iterable[Tuple[Iterable[Iterable[int]], str, float]],
    cfg: TokenizerConfig,
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    db = load_token_db(cfg.db_path)
    token_to_text = {value["token"]: key for key, value in db.items()}

    output = image.copy()
    for bbox, text, _conf in ocr_results:
        token = text.strip()
        original = token_to_text.get(token)
        if not original:
            continue

        xs = [int(point[0]) for point in bbox]
        ys = [int(point[1]) for point in bbox]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        if x_max <= x_min or y_max <= y_min:
            continue

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), background_color, -1)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        box_w = x_max - x_min
        box_h = y_max - y_min
        font_scale, (tw, th) = _fit_text(original, box_w, box_h, font_face, thickness)
        if font_scale <= 0:
            continue
        x = x_min + max(2, (box_w - tw) // 2)
        y = y_min + max(2, (box_h + th) // 2)
        cv2.putText(output, original, (x, y), font_face, font_scale, text_color, thickness)

    return output


def restore_image_from_token_map(
    image: np.ndarray,
    entries: List[Dict[str, Any]],
    background_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    output = image.copy()
    for item in entries:
        original = item.get("original", "")
        bbox = item.get("bbox", [])
        if not original or len(bbox) != 4:
            continue
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            continue
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), background_color, -1)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        box_w = x_max - x_min
        box_h = y_max - y_min
        font_scale, (tw, th) = _fit_text(original, box_w, box_h, font_face, thickness)
        if font_scale <= 0:
            continue
        x = x_min + max(2, (box_w - tw) // 2)
        y = y_min + max(2, (box_h + th) // 2)
        cv2.putText(output, original, (x, y), font_face, font_scale, text_color, thickness)
    return output
