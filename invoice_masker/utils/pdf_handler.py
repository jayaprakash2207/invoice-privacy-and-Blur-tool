import os
from typing import List, Optional

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


from utils.config import PDFConfig

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


def load_images_from_path(path: str, pdf_config: Optional[PDFConfig] = None) -> List[np.ndarray]:
    if not os.path.exists(path):
        return []

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return _load_from_pdf(path, pdf_config)

    if ext in SUPPORTED_IMAGE_EXTS:
        image = cv2.imread(path)
        return [image] if image is not None else []

    return []


def _load_from_pdf(path: str, pdf_config: Optional[PDFConfig]) -> List[np.ndarray]:
    try:
        if pdf_config:
            pages = convert_from_path(
                path,
                dpi=pdf_config.dpi,
                thread_count=pdf_config.thread_count,
            )
        else:
            pages = convert_from_path(path)
    except Exception:
        return []

    images: List[np.ndarray] = []
    for page in pages:
        if isinstance(page, Image.Image):
            np_page = np.array(page)
            if np_page.ndim == 2:
                np_page = cv2.cvtColor(np_page, cv2.COLOR_GRAY2BGR)
            elif np_page.shape[2] == 4:
                np_page = cv2.cvtColor(np_page, cv2.COLOR_RGBA2BGR)
            else:
                np_page = cv2.cvtColor(np_page, cv2.COLOR_RGB2BGR)
            images.append(np_page)
    return images
