from typing import Tuple

import cv2
import numpy as np


def add_text_watermark(
    image: np.ndarray,
    text: str,
    opacity: float = 0.25,
    font_scale: float = 1.2,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    if not text:
        return image

    overlay = image.copy()
    h, w = overlay.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(10, (w - tw) // 2)
    y = max(th + 10, (h - th) // 2)

    cv2.putText(overlay, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)
