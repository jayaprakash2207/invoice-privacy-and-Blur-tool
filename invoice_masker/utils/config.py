from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

from ner.detector import NERBackend


@dataclass
class PreprocessConfig:
    enabled: bool = True
    upscale_max_width: int = 1800
    downscale_max_width: int = 2000
    clahe_clip: float = 2.0
    denoise: bool = True
    sharpen: bool = False


@dataclass
class OCRConfig:
    languages: List[str] = field(default_factory=lambda: ["en"])
    gpu: bool = False
    text_threshold: float = 0.7
    low_text: float = 0.4
    link_threshold: float = 0.4
    canvas_size: int = 2560
    mag_ratio: float = 1.5
    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


@dataclass
class NERConfig:
    backend: NERBackend = NERBackend.SPACY
    spacy_model: str = "en_core_web_sm"
    hf_model: str = "dslim/bert-base-NER"


@dataclass
class MaskingConfig:
    padding: int = 2
    blur_divisor: int = 7
    min_kernel: int = 3


@dataclass
class PDFConfig:
    dpi: int = 250
    thread_count: int = 2


@dataclass
class RedactionConfig:
    smart_faker: bool = False
    charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    min_len: int = 4


@dataclass
class TokenizationConfig:
    output_dir: str = "outputs_tokenized"
    db_path: str = "outputs_tokenized/token_map.json"
    restore_output_dir: str = "outputs_restored"
    prefix_map: dict = field(
        default_factory=lambda: {
            "PERSON": "USER_",
            "ORG": "ORG_",
            "GPE": "LOC_",
            "EMAIL": "EMAIL_",
            "PHONE": "PHONE_",
            "ADDRESS": "ADDR_",
            "ZIP": "ZIP_",
            "DATE": "DATE_",
            "INVOICE": "INV_",
            "GEN": "GEN_",
        }
    )


@dataclass
class ClassificationConfig:
    enabled: bool = False
    model_name: str = "facebook/bart-large-mnli"
    candidate_labels: list[str] = field(
        default_factory=lambda: [
            "personal data",
            "financial data",
            "address or contact",
            "non-sensitive",
        ]
    )
    threshold: float = 0.6
    mode: str = "and"  # and/or: combine classifier with ner/regex hits


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "logs/invoice_masker.log"
    console: bool = True


@dataclass
class AppConfig:
    outputs_dir: str = "outputs"
    outputs_ocr_dir: str = "outputs_ocr"
    outputs_steps_dir: str = "outputs_steps"
    outputs_redacted_dir: str = "outputs_redacted"
    ocr: OCRConfig = field(default_factory=OCRConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    pdf: PDFConfig = field(default_factory=PDFConfig)
    redaction: RedactionConfig = field(default_factory=RedactionConfig)
    tokenization: TokenizationConfig = field(default_factory=TokenizationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    watermark_enabled: bool = True
    watermark_text: str = "CONFIDENTIAL - DEMO COPY"
    watermark_opacity: float = 0.25
    watermark_font_scale: float = 1.2
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dict(default: dict, override: dict) -> dict:
    result = dict(default)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | None = None) -> AppConfig:
    env_path = os.getenv("INVOICE_MASKER_CONFIG")
    path = Path(config_path or env_path or Path(__file__).resolve().parent.parent / "config.yaml")

    base_config = AppConfig()
    if not path.exists():
        return base_config

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    merged = _merge_dict(_to_dict(base_config), raw)
    return _from_dict(merged)


def _to_dict(cfg: AppConfig) -> dict:
    return {
        "outputs_dir": cfg.outputs_dir,
        "outputs_ocr_dir": cfg.outputs_ocr_dir,
        "outputs_steps_dir": cfg.outputs_steps_dir,
        "outputs_redacted_dir": cfg.outputs_redacted_dir,
        "ocr": {
            "languages": cfg.ocr.languages,
            "gpu": cfg.ocr.gpu,
            "text_threshold": cfg.ocr.text_threshold,
            "low_text": cfg.ocr.low_text,
            "link_threshold": cfg.ocr.link_threshold,
            "canvas_size": cfg.ocr.canvas_size,
            "mag_ratio": cfg.ocr.mag_ratio,
            "contrast_ths": cfg.ocr.contrast_ths,
            "adjust_contrast": cfg.ocr.adjust_contrast,
            "preprocess": {
                "enabled": cfg.ocr.preprocess.enabled,
                "upscale_max_width": cfg.ocr.preprocess.upscale_max_width,
                "downscale_max_width": cfg.ocr.preprocess.downscale_max_width,
                "clahe_clip": cfg.ocr.preprocess.clahe_clip,
                "denoise": cfg.ocr.preprocess.denoise,
                "sharpen": cfg.ocr.preprocess.sharpen,
            },
        },
        "ner": {
            "backend": cfg.ner.backend.value,
            "spacy_model": cfg.ner.spacy_model,
            "hf_model": cfg.ner.hf_model,
        },
        "masking": {
            "padding": cfg.masking.padding,
            "blur_divisor": cfg.masking.blur_divisor,
            "min_kernel": cfg.masking.min_kernel,
        },
        "pdf": {
            "dpi": cfg.pdf.dpi,
            "thread_count": cfg.pdf.thread_count,
        },
        "redaction": {
            "smart_faker": cfg.redaction.smart_faker,
            "charset": cfg.redaction.charset,
            "min_len": cfg.redaction.min_len,
        },
        "tokenization": {
            "output_dir": cfg.tokenization.output_dir,
            "db_path": cfg.tokenization.db_path,
            "restore_output_dir": cfg.tokenization.restore_output_dir,
            "prefix_map": cfg.tokenization.prefix_map,
        },
        "classification": {
            "enabled": cfg.classification.enabled,
            "model_name": cfg.classification.model_name,
            "candidate_labels": cfg.classification.candidate_labels,
            "threshold": cfg.classification.threshold,
            "mode": cfg.classification.mode,
        },
        "watermark": {
            "enabled": cfg.watermark_enabled,
            "text": cfg.watermark_text,
            "opacity": cfg.watermark_opacity,
            "font_scale": cfg.watermark_font_scale,
        },
        "logging": {
            "level": cfg.logging.level,
            "file": cfg.logging.file,
            "console": cfg.logging.console,
        },
    }


def _from_dict(data: dict) -> AppConfig:
    ocr_raw = data.get("ocr", {})
    preprocess_raw = ocr_raw.get("preprocess", {})
    ocr_cfg = OCRConfig(
        languages=ocr_raw.get("languages", ["en"]),
        gpu=ocr_raw.get("gpu", False),
        text_threshold=ocr_raw.get("text_threshold", 0.7),
        low_text=ocr_raw.get("low_text", 0.4),
        link_threshold=ocr_raw.get("link_threshold", 0.4),
        canvas_size=ocr_raw.get("canvas_size", 2560),
        mag_ratio=ocr_raw.get("mag_ratio", 1.5),
        contrast_ths=ocr_raw.get("contrast_ths", 0.1),
        adjust_contrast=ocr_raw.get("adjust_contrast", 0.5),
        preprocess=PreprocessConfig(
            enabled=preprocess_raw.get("enabled", True),
            upscale_max_width=preprocess_raw.get("upscale_max_width", 1800),
            downscale_max_width=preprocess_raw.get("downscale_max_width", 2000),
            clahe_clip=preprocess_raw.get("clahe_clip", 2.0),
            denoise=preprocess_raw.get("denoise", True),
            sharpen=preprocess_raw.get("sharpen", False),
        ),
    )

    ner_raw = data.get("ner", {})
    backend_value = ner_raw.get("backend", NERBackend.SPACY.value)
    ner_cfg = NERConfig(
        backend=NERBackend(backend_value),
        spacy_model=ner_raw.get("spacy_model", "en_core_web_sm"),
        hf_model=ner_raw.get("hf_model", "dslim/bert-base-NER"),
    )

    masking_raw = data.get("masking", {})
    masking_cfg = MaskingConfig(
        padding=masking_raw.get("padding", 2),
        blur_divisor=masking_raw.get("blur_divisor", 7),
        min_kernel=masking_raw.get("min_kernel", 3),
    )

    pdf_raw = data.get("pdf", {})
    pdf_cfg = PDFConfig(
        dpi=pdf_raw.get("dpi", 250),
        thread_count=pdf_raw.get("thread_count", 2),
    )

    redaction_raw = data.get("redaction", {})
    redaction_cfg = RedactionConfig(
        smart_faker=redaction_raw.get("smart_faker", False),
        charset=redaction_raw.get("charset", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
        min_len=redaction_raw.get("min_len", 4),
    )

    token_raw = data.get("tokenization", {})
    token_cfg = TokenizationConfig(
        output_dir=token_raw.get("output_dir", "outputs_tokenized"),
        db_path=token_raw.get("db_path", "outputs_tokenized/token_map.json"),
        restore_output_dir=token_raw.get("restore_output_dir", "outputs_restored"),
        prefix_map=token_raw.get(
            "prefix_map",
            {
                "PERSON": "USER_",
                "ORG": "ORG_",
                "GPE": "LOC_",
                "EMAIL": "EMAIL_",
                "PHONE": "PHONE_",
                "ADDRESS": "ADDR_",
                "ZIP": "ZIP_",
                "DATE": "DATE_",
                "INVOICE": "INV_",
                "GEN": "GEN_",
            },
        ),
    )

    class_raw = data.get("classification", {})
    class_cfg = ClassificationConfig(
        enabled=class_raw.get("enabled", False),
        model_name=class_raw.get("model_name", "facebook/bart-large-mnli"),
        candidate_labels=class_raw.get(
            "candidate_labels",
            ["personal data", "financial data", "address or contact", "non-sensitive"],
        ),
        threshold=class_raw.get("threshold", 0.6),
        mode=class_raw.get("mode", "and"),
    )

    watermark_raw = data.get("watermark", {})
    watermark_enabled = watermark_raw.get("enabled", True)
    watermark_text = watermark_raw.get("text", "CONFIDENTIAL - DEMO COPY")
    watermark_opacity = watermark_raw.get("opacity", 0.25)
    watermark_font_scale = watermark_raw.get("font_scale", 1.2)

    logging_raw = data.get("logging", {})
    logging_cfg = LoggingConfig(
        level=logging_raw.get("level", "INFO"),
        file=logging_raw.get("file", "logs/invoice_masker.log"),
        console=logging_raw.get("console", True),
    )

    return AppConfig(
        outputs_dir=data.get("outputs_dir", "outputs"),
        outputs_ocr_dir=data.get("outputs_ocr_dir", "outputs_ocr"),
        outputs_steps_dir=data.get("outputs_steps_dir", "outputs_steps"),
        outputs_redacted_dir=data.get("outputs_redacted_dir", "outputs_redacted"),
        ocr=ocr_cfg,
        ner=ner_cfg,
        masking=masking_cfg,
        pdf=pdf_cfg,
        redaction=redaction_cfg,
        tokenization=token_cfg,
        classification=class_cfg,
        watermark_enabled=watermark_enabled,
        watermark_text=watermark_text,
        watermark_opacity=watermark_opacity,
        watermark_font_scale=watermark_font_scale,
        logging=logging_cfg,
    )
