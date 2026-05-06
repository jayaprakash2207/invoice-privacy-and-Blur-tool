import os
import shutil
import tempfile
import uuid

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from main import mask_images
from ner.detector import NERBackend
from utils.config import load_config
from utils.logger import setup_logging
from utils.pdf_handler import load_images_from_path
from utils.watermark import add_text_watermark
from utils.tokenizer import (
    restore_image_from_tokens,
    restore_image_from_tokens_ocr,
    restore_image_from_token_map,
    TokenizerConfig,
)


app = FastAPI(title="Invoice Privacy Masker")
CONFIG = load_config()
setup_logging(CONFIG.logging)
templates = Jinja2Templates(directory="templates")

app.mount("/outputs", StaticFiles(directory=CONFIG.outputs_dir), name="outputs")
app.mount("/outputs_ocr", StaticFiles(directory=CONFIG.outputs_ocr_dir), name="outputs_ocr")
app.mount("/outputs_steps", StaticFiles(directory=CONFIG.outputs_steps_dir), name="outputs_steps")
app.mount("/outputs_redacted", StaticFiles(directory=CONFIG.outputs_redacted_dir), name="outputs_redacted")
app.mount("/outputs_tokenized", StaticFiles(directory=CONFIG.tokenization.output_dir), name="outputs_tokenized")
app.mount("/outputs_restored", StaticFiles(directory=CONFIG.tokenization.restore_output_dir), name="outputs_restored")


def _cleanup(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "error": None,
            "restore_results": None,
            "restore_error": None,
        },
    )


@app.post("/restore")
async def restore_tokenized(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    temp_dir = tempfile.mkdtemp(prefix="invoice_masker_")
    input_path = os.path.join(temp_dir, file.filename)

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    images = load_images_from_path(input_path, CONFIG.pdf)
    if not images:
        _cleanup(temp_dir)
        raise HTTPException(status_code=400, detail="Unsupported or invalid file.")

    base_name = os.path.splitext(os.path.basename(file.filename))[0]
    token_cfg = TokenizerConfig(
        output_dir=CONFIG.tokenization.output_dir,
        db_path=CONFIG.tokenization.db_path,
        prefix_map=CONFIG.tokenization.prefix_map,
    )

    outputs = []
    from main import _rescale_ocr_results
    from main import _load_token_map_for_image
    from ocr.ocr_engine import EasyOCREngine
    from utils.config import OCRConfig, PreprocessConfig

    restore_ocr = OCRConfig(
        languages=CONFIG.ocr.languages,
        gpu=CONFIG.ocr.gpu,
        text_threshold=CONFIG.ocr.text_threshold,
        low_text=CONFIG.ocr.low_text,
        link_threshold=CONFIG.ocr.link_threshold,
        canvas_size=1280,
        mag_ratio=1.0,
        contrast_ths=CONFIG.ocr.contrast_ths,
        adjust_contrast=CONFIG.ocr.adjust_contrast,
        preprocess=PreprocessConfig(enabled=False),
    )
    ocr_engine = EasyOCREngine(restore_ocr)

    for page_idx, image in enumerate(images, start=1):
        map_entries = _load_token_map_for_image(input_path, token_cfg.output_dir)
        if map_entries:
            restored = restore_image_from_token_map(image, map_entries)
        else:
            ocr_results, scale = ocr_engine.extract(image)
            if scale != 1.0:
                ocr_results = _rescale_ocr_results(ocr_results, scale)
            restored = restore_image_from_tokens_ocr(image, ocr_results, token_cfg)
        restored = add_text_watermark(
            restored,
            text=CONFIG.watermark_text,
            opacity=CONFIG.watermark_opacity,
            font_scale=CONFIG.watermark_font_scale,
        )
        suffix = f"_page_{page_idx}" if len(images) > 1 else ""
        out_path = os.path.join(
            CONFIG.tokenization.restore_output_dir,
            f"{base_name}{suffix}_restored.png",
        )
        os.makedirs(CONFIG.tokenization.restore_output_dir, exist_ok=True)
        cv2.imwrite(out_path, restored)
        outputs.append(out_path)

    background_tasks.add_task(_cleanup, temp_dir)
    return FileResponse(
        outputs[0],
        media_type="image/png",
        filename=os.path.basename(outputs[0]),
        headers={"X-Page-Count": str(len(outputs))},
    )


@app.post("/mask")
async def mask_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ner_backend: NERBackend = NERBackend.SPACY,
    save_ocr_boxes: bool = True,
    save_steps: bool = False,
    save_dummy: bool = False,
    smart_faker: bool = False,
    save_tokenized: bool = False,
    apply_watermark: bool = True,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    temp_dir = tempfile.mkdtemp(prefix="invoice_masker_")
    input_path = os.path.join(temp_dir, file.filename)

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    images = load_images_from_path(input_path, CONFIG.pdf)
    if not images:
        _cleanup(temp_dir)
        raise HTTPException(status_code=400, detail="Unsupported or invalid file.")

    base_name = os.path.splitext(os.path.basename(file.filename))[0]
    unique_suffix = uuid.uuid4().hex[:8]
    base_name = f"{base_name}_{unique_suffix}"
    outputs = mask_images(
        images,
        CONFIG.outputs_dir,
        CONFIG.outputs_ocr_dir,
        CONFIG.outputs_steps_dir,
        CONFIG.outputs_redacted_dir,
        ner_backend,
        base_name,
        CONFIG,
        save_ocr_boxes=save_ocr_boxes,
        save_steps=save_steps,
        save_dummy=save_dummy,
        smart_faker=smart_faker or CONFIG.redaction.smart_faker,
        save_tokenized=save_tokenized,
        apply_watermark=apply_watermark and CONFIG.watermark_enabled,
    )

    if not outputs:
        _cleanup(temp_dir)
        raise HTTPException(status_code=500, detail="Failed to produce output.")

    # If PDF, return the first masked page to keep response simple.
    output_path = outputs[0]

    background_tasks.add_task(_cleanup, temp_dir)
    return FileResponse(
        output_path,
        media_type="image/png",
        filename=os.path.basename(output_path),
        headers={"X-Page-Count": str(len(outputs))},
    )


@app.post("/upload")
async def upload_ui(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    save_ocr_boxes: str | None = Form(None),
    save_steps: str | None = Form(None),
    save_dummy: str | None = Form(None),
    smart_faker: str | None = Form(None),
    save_tokenized: str | None = Form(None),
    apply_watermark: str | None = Form(None),
):
    try:
        flags = {
            "save_ocr_boxes": bool(save_ocr_boxes),
            "save_steps": bool(save_steps),
            "save_dummy": bool(save_dummy),
            "smart_faker": bool(smart_faker),
            "save_tokenized": bool(save_tokenized),
            "apply_watermark": bool(apply_watermark),
        }

        temp_dir = tempfile.mkdtemp(prefix="invoice_masker_")
        input_path = os.path.join(temp_dir, file.filename)

        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        images = load_images_from_path(input_path, CONFIG.pdf)
        if not images:
            _cleanup(temp_dir)
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "results": None, "error": "Unsupported or invalid file."},
            )

        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        unique_suffix = uuid.uuid4().hex[:8]
        base_name = f"{base_name}_{unique_suffix}"

        mask_images(
            images,
            CONFIG.outputs_dir,
            CONFIG.outputs_ocr_dir,
            CONFIG.outputs_steps_dir,
            CONFIG.outputs_redacted_dir,
            NERBackend(CONFIG.ner.backend.value),
            base_name,
            CONFIG,
            save_ocr_boxes=flags["save_ocr_boxes"],
            save_steps=flags["save_steps"],
            save_dummy=flags["save_dummy"],
            smart_faker=flags["smart_faker"],
            save_tokenized=flags["save_tokenized"],
            apply_watermark=flags["apply_watermark"] and CONFIG.watermark_enabled,
        )

        results = [
            {
                "label": f"Masked Output: {base_name}_masked.png",
                "url": f"/outputs/{base_name}_masked.png",
            }
        ]
        if flags["save_ocr_boxes"]:
            results.append(
                {
                    "label": f"OCR Boxes: {base_name}_ocr_boxes.png",
                    "url": f"/outputs_ocr/{base_name}_ocr_boxes.png",
                }
            )
            results.append(
                {
                    "label": f"OCR JSON: {base_name}_ocr_boxes.json",
                    "url": f"/outputs_ocr/{base_name}_ocr_boxes.json",
                }
            )
        if flags["save_steps"]:
            results.append(
                {
                    "label": f"Step4 Sensitive Overlay: {base_name}.png",
                    "url": f"/outputs_steps/step4_sensitive/{base_name}.png",
                }
            )
            results.append(
                {
                    "label": f"Step5 Masked Preview: {base_name}.png",
                    "url": f"/outputs_steps/step5_masked/{base_name}.png",
                }
            )
        if flags["save_dummy"]:
            results.append(
                {
                    "label": f"Redacted Output: {base_name}_redacted.png",
                    "url": f"/outputs_redacted/{base_name}_redacted.png",
                }
            )
        if flags["save_tokenized"]:
            results.append(
                {
                    "label": f"Tokenized Output: {base_name}_tokenized.png",
                    "url": f"/outputs_tokenized/{base_name}_tokenized.png",
                }
            )
            results.append(
                {
                    "label": f"Token Map JSON: {base_name}_tokenized_map.json",
                    "url": f"/outputs_tokenized/{base_name}_tokenized_map.json",
                }
            )

        background_tasks.add_task(_cleanup, temp_dir)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": results,
                "error": None,
                "restore_results": None,
                "restore_error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "error": f"Upload failed: {exc}",
                "restore_results": None,
                "restore_error": None,
            },
        )


@app.post("/restore_upload")
async def restore_upload_ui(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    try:
        temp_dir = tempfile.mkdtemp(prefix="invoice_masker_")
        input_path = os.path.join(temp_dir, file.filename)

        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        images = load_images_from_path(input_path, CONFIG.pdf)
        if not images:
            _cleanup(temp_dir)
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "results": None,
                    "error": None,
                    "restore_results": None,
                    "restore_error": "Unsupported or invalid file.",
                },
            )

        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        unique_suffix = uuid.uuid4().hex[:8]
        base_name = f"{base_name}_{unique_suffix}"

        token_cfg = TokenizerConfig(
            output_dir=CONFIG.tokenization.output_dir,
            db_path=CONFIG.tokenization.db_path,
            prefix_map=CONFIG.tokenization.prefix_map,
        )

        from main import _rescale_ocr_results, _load_token_map_for_image
        from ocr.ocr_engine import EasyOCREngine
        from utils.config import OCRConfig, PreprocessConfig

        restore_ocr = OCRConfig(
            languages=CONFIG.ocr.languages,
            gpu=CONFIG.ocr.gpu,
            text_threshold=CONFIG.ocr.text_threshold,
            low_text=CONFIG.ocr.low_text,
            link_threshold=CONFIG.ocr.link_threshold,
            canvas_size=1280,
            mag_ratio=1.0,
            contrast_ths=CONFIG.ocr.contrast_ths,
            adjust_contrast=CONFIG.ocr.adjust_contrast,
            preprocess=PreprocessConfig(enabled=False),
        )
        ocr_engine = EasyOCREngine(restore_ocr)

        outputs = []
        for page_idx, image in enumerate(images, start=1):
            map_entries = _load_token_map_for_image(input_path, token_cfg.output_dir)
            if map_entries:
                restored = restore_image_from_token_map(image, map_entries)
            else:
                ocr_results, scale = ocr_engine.extract(image)
                if scale != 1.0:
                    ocr_results = _rescale_ocr_results(ocr_results, scale)
                restored = restore_image_from_tokens_ocr(image, ocr_results, token_cfg)
            if CONFIG.watermark_enabled:
                restored = add_text_watermark(
                    restored,
                    text=CONFIG.watermark_text,
                    opacity=CONFIG.watermark_opacity,
                    font_scale=CONFIG.watermark_font_scale,
                )
            suffix = f"_page_{page_idx}" if len(images) > 1 else ""
            out_path = os.path.join(
                CONFIG.tokenization.restore_output_dir,
                f"{base_name}{suffix}_restored.png",
            )
            os.makedirs(CONFIG.tokenization.restore_output_dir, exist_ok=True)
            cv2.imwrite(out_path, restored)
            outputs.append(out_path)

        restore_results = [
            {
                "label": os.path.basename(path),
                "url": f"/outputs_restored/{os.path.basename(path)}",
            }
            for path in outputs
        ]

        background_tasks.add_task(_cleanup, temp_dir)
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "error": None,
                "restore_results": restore_results,
                "restore_error": None,
            },
        )
    except Exception as exc:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "error": None,
                "restore_results": None,
                "restore_error": f"Restore failed: {exc}",
            },
        )
