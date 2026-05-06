from typing import Iterable

import cv2
import numpy as np


def blur_regions(
    image: np.ndarray,
    boxes: Iterable[Iterable[int]],
    padding: int = 2,
    blur_divisor: int = 7,
    min_kernel: int = 3,
) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    output = image.copy()
    height, width = output.shape[:2]

    for box in boxes:
        if len(box) != 4:
            continue
        x_min, y_min, x_max, y_max = box

        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)

        if x_max <= x_min or y_max <= y_min:
            continue

        roi = output[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue

        kx = max(min_kernel, ((x_max - x_min) // blur_divisor) * 2 + 1)
        ky = max(min_kernel, ((y_max - y_min) // blur_divisor) * 2 + 1)

        blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
        output[y_min:y_max, x_min:x_max] = blurred

    return output
