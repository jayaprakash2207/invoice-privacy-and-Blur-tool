from functools import lru_cache
from typing import List, Tuple

import cv2
import easyocr
import numpy as np

from utils.config import OCRConfig


@lru_cache(maxsize=2)
def _get_reader(languages: Tuple[str, ...], gpu: bool) -> easyocr.Reader:
    return easyocr.Reader(list(languages), gpu=gpu)


class EasyOCREngine:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self.reader = _get_reader(tuple(config.languages), config.gpu)

    def extract(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[List[List[int]], str, float]], float]:
        if image is None or image.size == 0:
            return [], 1.0

        processed, scale = self._preprocess(image)
        try:
            results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                text_threshold=self.config.text_threshold,
                low_text=self.config.low_text,
                link_threshold=self.config.link_threshold,
                canvas_size=self.config.canvas_size,
                mag_ratio=self.config.mag_ratio,
                contrast_ths=self.config.contrast_ths,
                adjust_contrast=self.config.adjust_contrast,
            )
            return (results or []), scale
        except RuntimeError as exc:
            if "not enough memory" not in str(exc).lower():
                raise

        # Fallback: downscale + smaller canvas to avoid OOM.
        fallback_scale = 1.0
        max_width = 1400
        if processed.shape[1] > max_width:
            fallback_scale = max_width / max(1, processed.shape[1])
            new_size = (
                int(processed.shape[1] * fallback_scale),
                int(processed.shape[0] * fallback_scale),
            )
            processed = cv2.resize(processed, new_size, interpolation=cv2.INTER_AREA)

        results = self.reader.readtext(
            processed,
            detail=1,
            paragraph=False,
            text_threshold=self.config.text_threshold,
            low_text=self.config.low_text,
            link_threshold=self.config.link_threshold,
            canvas_size=min(self.config.canvas_size, 1280),
            mag_ratio=min(self.config.mag_ratio, 1.0),
            contrast_ths=self.config.contrast_ths,
            adjust_contrast=self.config.adjust_contrast,
        )
        return (results or []), scale * fallback_scale

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        cfg = self.config.preprocess
        if not cfg.enabled:
            return image, 1.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cfg.denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=15)

        if cfg.clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        if cfg.sharpen:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)

        scale = 1.0
        width = gray.shape[1]
        if cfg.downscale_max_width and width > cfg.downscale_max_width:
            scale = cfg.downscale_max_width / max(1, width)
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
        elif cfg.upscale_max_width and width < cfg.upscale_max_width:
            scale = cfg.upscale_max_width / max(1, width)
            new_size = (int(gray.shape[1] * scale), int(gray.shape[0] * scale))
            gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_CUBIC)

        return gray, scale
