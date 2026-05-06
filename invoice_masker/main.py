import argparse
import json
import logging
import os
import sys
from typing import List

import cv2
import numpy as np

from ocr.ocr_engine import EasyOCREngine
from ner.detector import (
    NERBackend,
    SensitiveDataDetector,
    SpacyNERDetector,
    HuggingFaceNERDetector,
)
from utils.blur import blur_regions
from utils.config import AppConfig, load_config, OCRConfig, PreprocessConfig
from utils.logger import setup_logging
from utils.pdf_handler import load_images_from_path
from utils.redact import redact_with_dummy_text
from utils.watermark import add_text_watermark
from utils.tokenizer import (
    tokenize_image,
    restore_image_from_tokens,
    restore_image_from_tokens_ocr,
    restore_image_from_token_map,
    TokenizerConfig,
)


LOGGER = logging.getLogger("invoice_masker")


def build_detector(config: AppConfig, backend: NERBackend) -> SensitiveDataDetector:
    if backend == NERBackend.SPACY:
        ner = SpacyNERDetector(config.ner.spacy_model)
    elif backend == NERBackend.HUGGINGFACE:
        ner = HuggingFaceNERDetector(config.ner.hf_model)
    else:
        raise ValueError(f"Unsupported NER backend: {backend}")
    classifier = None
    if config.classification.enabled:
        from ner.detector import TextClassifier

        try:
            classifier = TextClassifier(
                model_name=config.classification.model_name,
                candidate_labels=config.classification.candidate_labels,
                threshold=config.classification.threshold,
            )
        except Exception as exc:
            LOGGER.warning(
                "Classifier disabled: failed to load model %s (%s)",
                config.classification.model_name,
                exc,
            )
            classifier = None
    return SensitiveDataDetector(
        ner,
        classifier=classifier,
        classifier_mode=config.classification.mode,
    )


def mask_images(
    images: List,
    output_dir: str,
    ocr_output_dir: str,
    steps_output_dir: str,
    redacted_output_dir: str,
    backend: NERBackend,
    base_name: str,
    config: AppConfig,
    save_ocr_boxes: bool = False,
    save_steps: bool = False,
    save_dummy: bool = False,
    smart_faker: bool = False,
    save_tokenized: bool = False,
    apply_watermark: bool = True,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    if save_ocr_boxes:
        os.makedirs(ocr_output_dir, exist_ok=True)
    if save_steps:
        os.makedirs(steps_output_dir, exist_ok=True)
    if save_dummy:
        os.makedirs(redacted_output_dir, exist_ok=True)

    ocr_engine = EasyOCREngine(config.ocr)
    detector = build_detector(config, backend)

    output_paths: List[str] = []

    for page_idx, image in enumerate(images, start=1):
        try:
            ocr_results, scale = ocr_engine.extract(image)
            if scale != 1.0:
                ocr_results = _rescale_ocr_results(ocr_results, scale)
            if save_steps or (save_dummy and smart_faker) or save_tokenized:
                boxes, detected_fields, details = detector.detect_sensitive_details(
                    ocr_results
                )
            else:
                boxes, detected_fields = detector.detect_sensitive_regions(ocr_results)
        except Exception as exc:
            LOGGER.exception("OCR/NER failed on page %s: %s", page_idx, exc)
            continue

        LOGGER.info(
            "Page %s: detected %s sensitive field(s).",
            page_idx,
            len(detected_fields),
        )
        if detected_fields:
            LOGGER.info("Detected fields: %s", ", ".join(sorted(detected_fields)))

        masked = blur_regions(
            image,
            boxes,
            padding=config.masking.padding,
            blur_divisor=config.masking.blur_divisor,
            min_kernel=config.masking.min_kernel,
        )
        if apply_watermark:
            masked = add_text_watermark(
                masked,
                text=config.watermark_text,
                opacity=config.watermark_opacity,
                font_scale=config.watermark_font_scale,
            )
        suffix = f"_page_{page_idx}" if len(images) > 1 else ""
        output_path = os.path.join(output_dir, f"{base_name}{suffix}_masked.png")
        if not cv2.imwrite(output_path, masked):
            LOGGER.error("Failed to write output file: %s", output_path)
            continue
        output_paths.append(output_path)

        if save_dummy:
            redacted = redact_with_dummy_text(
                image,
                ocr_results,
                boxes,
                details=details if (save_steps or smart_faker) else None,
                smart_faker=smart_faker,
                charset=config.redaction.charset,
                min_len=config.redaction.min_len,
            )
            if apply_watermark:
                redacted = add_text_watermark(
                    redacted,
                    text=config.watermark_text,
                    opacity=config.watermark_opacity,
                    font_scale=config.watermark_font_scale,
                )
            redacted_path = os.path.join(
                redacted_output_dir, f"{base_name}{suffix}_redacted.png"
            )
            if not cv2.imwrite(redacted_path, redacted):
                LOGGER.error("Failed to write redacted file: %s", redacted_path)

        if save_tokenized:
            token_cfg = TokenizerConfig(
                output_dir=config.tokenization.output_dir,
                db_path=config.tokenization.db_path,
                prefix_map=config.tokenization.prefix_map,
            )
            tokenized, _db, entries = tokenize_image(image, details, token_cfg)
            if apply_watermark:
                tokenized = add_text_watermark(
                    tokenized,
                    text=config.watermark_text,
                    opacity=config.watermark_opacity,
                    font_scale=config.watermark_font_scale,
                )
            tokenized_path = os.path.join(
                token_cfg.output_dir, f"{base_name}{suffix}_tokenized.png"
            )
            os.makedirs(token_cfg.output_dir, exist_ok=True)
            if not cv2.imwrite(tokenized_path, tokenized):
                LOGGER.error("Failed to write tokenized file: %s", tokenized_path)
            token_map_path = os.path.join(
                token_cfg.output_dir, f"{base_name}{suffix}_tokenized_map.json"
            )
            _save_token_map(entries, token_map_path)

        if save_ocr_boxes:
            ocr_json_path = os.path.join(
                ocr_output_dir, f"{base_name}{suffix}_ocr_boxes.json"
            )
            _save_ocr_boxes(ocr_results, ocr_json_path)
            ocr_image_path = os.path.join(
                ocr_output_dir, f"{base_name}{suffix}_ocr_boxes.png"
            )
            _save_ocr_boxes_image(image, ocr_results, ocr_image_path)

        if save_steps:
            _save_step_outputs(
                image=image,
                ocr_results=ocr_results,
                details=details,
                output_dir=steps_output_dir,
                base_name=base_name,
                suffix=suffix,
                watermark_text=config.watermark_text,
                watermark_opacity=config.watermark_opacity,
                watermark_font_scale=config.watermark_font_scale,
                apply_watermark=apply_watermark,
            )

    return output_paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoice Privacy Masker")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to an invoice image (PNG/JPG) or PDF.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save masked images.",
    )
    parser.add_argument(
        "--ner-backend",
        choices=[backend.value for backend in NERBackend],
        default=None,
        help="NER backend to use: spacy or huggingface.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config YAML. Overrides default config.yaml.",
    )
    parser.add_argument(
        "--save-ocr-boxes",
        action="store_true",
        help="Save OCR bounding boxes JSON + image in outputs_ocr directory.",
    )
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="Save per-step debug outputs in outputs_steps directory.",
    )
    parser.add_argument(
        "--save-dummy",
        action="store_true",
        help="Save redacted output with dummy text in outputs_redacted directory.",
    )
    parser.add_argument(
        "--smart-faker",
        action="store_true",
        help="Use Faker to generate realistic dummy data for redaction.",
    )
    parser.add_argument(
        "--save-tokenized",
        action="store_true",
        help="Save reversible tokenized output in outputs_tokenized directory.",
    )
    parser.add_argument(
        "--no-watermark",
        action="store_true",
        help="Disable watermark on outputs.",
    )
    parser.add_argument(
        "--restore-tokenized",
        action="store_true",
        help="Restore tokenized output using token map into outputs_restored.",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace, config: AppConfig) -> None:
    input_path = args.input
    backend = NERBackend(args.ner_backend or config.ner.backend.value)
    output_dir = args.output_dir or config.outputs_dir
    ocr_output_dir = config.outputs_ocr_dir
    steps_output_dir = config.outputs_steps_dir
    redacted_output_dir = config.outputs_redacted_dir

    if not os.path.exists(input_path):
        LOGGER.error("Input file not found: %s", input_path)
        sys.exit(1)

    images = load_images_from_path(input_path, config.pdf)
    if not images:
        LOGGER.error("No images could be loaded from input: %s", input_path)
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    outputs = mask_images(
        images,
        output_dir,
        ocr_output_dir,
        steps_output_dir,
        redacted_output_dir,
        backend,
        base_name,
        config,
        save_ocr_boxes=args.save_ocr_boxes,
        save_steps=args.save_steps,
        save_dummy=args.save_dummy,
        smart_faker=args.smart_faker or config.redaction.smart_faker,
        save_tokenized=args.save_tokenized,
        apply_watermark=not args.no_watermark and config.watermark_enabled,
    )
    if not outputs:
        LOGGER.error("No outputs produced. Check logs for details.")
        sys.exit(1)

    LOGGER.info("Masked files saved:")
    for path in outputs:
        LOGGER.info(" - %s", path)


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    setup_logging(config.logging)
    try:
        restore_only = (
            args.restore_tokenized
            and not args.save_ocr_boxes
            and not args.save_steps
            and not args.save_dummy
            and not args.save_tokenized
        )
        if restore_only:
            _run_restore(args, config)
        else:
            run_cli(args, config)
            if args.restore_tokenized:
                _run_restore(args, config)
    except Exception as exc:
        LOGGER.exception("Fatal error: %s", exc)
        sys.exit(1)


def _save_ocr_boxes(ocr_results, output_path: str) -> None:
    payload = []
    for bbox, text, conf in ocr_results:
        payload.append(
            {
                "text": text,
                "confidence": float(conf),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            }
        )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _save_token_map(entries: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _save_ocr_boxes_image(image, ocr_results, output_path: str) -> None:
    canvas = image.copy()
    for bbox, _text, _conf in ocr_results:
        points = [[int(p[0]), int(p[1])] for p in bbox]
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(
            canvas,
            [pts],
            isClosed=True,
            color=(0, 0, 255),
            thickness=2,
        )
    cv2.imwrite(output_path, canvas)


def _save_step_outputs(
    image,
    ocr_results,
    details,
    output_dir: str,
    base_name: str,
    suffix: str,
    watermark_text: str,
    watermark_opacity: float,
    watermark_font_scale: float,
    apply_watermark: bool,
) -> None:
    step2_dir = os.path.join(output_dir, "step2_text")
    step3_dir = os.path.join(output_dir, "step3_ner_regex")
    step4_dir = os.path.join(output_dir, "step4_sensitive")
    step5_dir = os.path.join(output_dir, "step5_masked")
    os.makedirs(step2_dir, exist_ok=True)
    os.makedirs(step3_dir, exist_ok=True)
    os.makedirs(step4_dir, exist_ok=True)
    os.makedirs(step5_dir, exist_ok=True)

    step2_path = os.path.join(step2_dir, f"{base_name}{suffix}.json")
    step3_path = os.path.join(step3_dir, f"{base_name}{suffix}.json")
    step4_path = os.path.join(step4_dir, f"{base_name}{suffix}.json")
    step4_img_path = os.path.join(step4_dir, f"{base_name}{suffix}.png")
    step5_img_path = os.path.join(step5_dir, f"{base_name}{suffix}.png")

    step2_payload = []
    for bbox, text, conf in ocr_results:
        step2_payload.append(
            {
                "text": text,
                "confidence": float(conf),
                "bbox": [[int(p[0]), int(p[1])] for p in bbox],
            }
        )
    with open(step2_path, "w", encoding="utf-8") as f:
        json.dump(step2_payload, f, ensure_ascii=False, indent=2)

    with open(step3_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2) 

    step4_payload = [
        {
            "text": item["text"],
            "bbox": item["bbox"],
            "sensitive": item["sensitive"],
        }
        for item in details
    ]
    with open(step4_path, "w", encoding="utf-8") as f:
        json.dump(step4_payload, f, ensure_ascii=False, indent=2)

    canvas = image.copy()
    for item in details:
        x_min, y_min, x_max, y_max = item["bbox"]
        color = (0, 0, 255) if item["sensitive"] else (0, 200, 0)
        cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.imwrite(step4_img_path, canvas)

    masked_preview = image.copy()
    for item in details:
        if not item["sensitive"]:
            continue
        x_min, y_min, x_max, y_max = item["bbox"]
        if x_max <= x_min or y_max <= y_min:
            continue
        roi = masked_preview[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            continue
        kx = max(3, ((x_max - x_min) // 7) * 2 + 1)
        ky = max(3, ((y_max - y_min) // 7) * 2 + 1)
        blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
        masked_preview[y_min:y_max, x_min:x_max] = blurred
    if apply_watermark:
        masked_preview = add_text_watermark(
            masked_preview,
            text=watermark_text,
            opacity=watermark_opacity,
            font_scale=watermark_font_scale,
        )
    cv2.imwrite(step5_img_path, masked_preview)


def _rescale_ocr_results(ocr_results, scale: float):
    rescaled = []
    for bbox, text, conf in ocr_results:
        new_bbox = [[int(p[0] / scale), int(p[1] / scale)] for p in bbox]
        rescaled.append((new_bbox, text, conf))
    return rescaled


def _load_token_map_for_image(input_path: str, token_output_dir: str) -> list[dict]:
    stem = os.path.splitext(os.path.basename(input_path))[0]
    candidates = [
        f"{stem}_tokenized_map.json",
        f"{stem}_map.json",
    ]
    if stem.endswith("_tokenized"):
        base = stem[: -len("_tokenized")]
        candidates.extend(
            [
                f"{base}_tokenized_map.json",
                f"{base}_map.json",
            ]
        )

    search_dirs = [os.path.dirname(input_path), token_output_dir]
    for directory in search_dirs:
        for name in candidates:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            return data
                except Exception:
                    continue
    return []


def _run_restore(args: argparse.Namespace, config: AppConfig) -> None:
    input_path = args.input
    if not os.path.exists(input_path):
        LOGGER.error("Input file not found for restore: %s", input_path)
        return

    images = load_images_from_path(input_path, config.pdf)
    if not images:
        LOGGER.error("No images could be loaded for restore: %s", input_path)
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    token_cfg = TokenizerConfig(
        output_dir=config.tokenization.output_dir,
        db_path=config.tokenization.db_path,
        prefix_map=config.tokenization.prefix_map,
    )

    restore_ocr = OCRConfig(
        languages=config.ocr.languages,
        gpu=config.ocr.gpu,
        text_threshold=config.ocr.text_threshold,
        low_text=config.ocr.low_text,
        link_threshold=config.ocr.link_threshold,
        canvas_size=1280,
        mag_ratio=1.0,
        contrast_ths=config.ocr.contrast_ths,
        adjust_contrast=config.ocr.adjust_contrast,
        preprocess=PreprocessConfig(enabled=False),
    )
    ocr_engine = EasyOCREngine(restore_ocr)

    os.makedirs(config.tokenization.restore_output_dir, exist_ok=True)

    for page_idx, image in enumerate(images, start=1):
        map_entries = _load_token_map_for_image(input_path, token_cfg.output_dir)
        if map_entries:
            restored = restore_image_from_token_map(image, map_entries)
        else:
            ocr_results, scale = ocr_engine.extract(image)
            if scale != 1.0:
                ocr_results = _rescale_ocr_results(ocr_results, scale)
            restored = restore_image_from_tokens_ocr(image, ocr_results, token_cfg)
        if config.watermark_enabled:
            restored = add_text_watermark(
                restored,
                text=config.watermark_text,
                opacity=config.watermark_opacity,
                font_scale=config.watermark_font_scale,
            )
        suffix = f"_page_{page_idx}" if len(images) > 1 else ""
        output_path = os.path.join(
            config.tokenization.restore_output_dir, f"{base_name}{suffix}_restored.png"
        )
        cv2.imwrite(output_path, restored)


if __name__ == "__main__":
    main()
