# Invoice Privacy & Blur Tool — Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Features](#3-features)
4. [Technology Stack](#4-technology-stack)
5. [Module Reference](#5-module-reference)
   - 5.1 [Entry Points](#51-entry-points)
   - 5.2 [OCR Engine](#52-ocr-engine)
   - 5.3 [NER & Sensitive-Data Detection](#53-ner--sensitive-data-detection)
   - 5.4 [Blur (Masking)](#54-blur-masking)
   - 5.5 [Redaction](#55-redaction)
   - 5.6 [Tokenizer](#56-tokenizer)
   - 5.7 [Watermark](#57-watermark)
   - 5.8 [PDF Handler](#58-pdf-handler)
   - 5.9 [Configuration](#59-configuration)
   - 5.10 [Logger](#510-logger)
6. [Processing Pipeline](#6-processing-pipeline)
7. [API Reference](#7-api-reference)
8. [CLI Reference](#8-cli-reference)
9. [Configuration Reference](#9-configuration-reference)
10. [Output Artefacts](#10-output-artefacts)
11. [Deployment](#11-deployment)
12. [Security & Privacy Design](#12-security--privacy-design)
13. [Testing](#13-testing)
14. [Directory Structure](#14-directory-structure)
15. [Glossary](#15-glossary)

---

## 1. Project Overview

**Invoice Privacy & Blur Tool** is an automated document-privacy solution designed to detect and obscure personally identifiable information (PII) and sensitive financial data found in invoice images and PDF files.

The tool uses a multi-stage pipeline combining Optical Character Recognition (OCR), Named-Entity Recognition (NER), and rule-based Regex detection to locate sensitive fields on a document and then applies one or more of the following privacy operations:

| Operation | Description |
|-----------|-------------|
| **Blur / Mask** | Gaussian blur over sensitive bounding boxes |
| **Redact** | Replaces sensitive text with random or realistic dummy values |
| **Tokenize** | Replaces sensitive text with reversible opaque tokens |
| **Restore** | Re-inserts original values into a previously tokenized image |

The solution is accessible via a **command-line interface (CLI)** and a **FastAPI web interface / REST API**, and can be deployed locally or inside a Docker container.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       Invoice Privacy Tool                       │
│                                                                  │
│  Input (PNG / JPG / PDF)                                         │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────┐                                                 │
│  │ PDF Handler │  Converts PDF pages → OpenCV images            │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                 │
│  │ OCR Engine  │  EasyOCR → list of (bbox, text, confidence)    │
│  │ (EasyOCR)   │  with optional image pre-processing            │
│  └──────┬──────┘                                                 │
│         │                                                        │
│         ▼                                                        │
│  ┌───────────────────────────────────┐                           │
│  │   Sensitive Data Detector         │                           │
│  │  ┌─────────┐  ┌───────────────┐  │                           │
│  │  │   NER   │  │ Regex Engine  │  │                           │
│  │  │(spaCy / │  │ (email, phone,│  │                           │
│  │  │  HF)    │  │  date, addr…) │  │                           │
│  │  └────┬────┘  └───────┬───────┘  │                           │
│  │       └──────┬─────────┘          │                           │
│  │         ┌────▼────────┐           │                           │
│  │         │  Optional   │           │                           │
│  │         │ Zero-Shot   │           │                           │
│  │         │ Classifier  │           │                           │
│  │         └────┬────────┘           │                           │
│  └──────────────┼───────────────────┘                           │
│                 │  sensitive boxes + labels                       │
│                 ▼                                                │
│  ┌──────────────────────────────────────────────┐               │
│  │            Output Operations                 │               │
│  │  ┌────────┐ ┌─────────┐ ┌────────┐          │               │
│  │  │  Blur  │ │ Redact  │ │Tokenize│          │               │
│  │  └────────┘ └─────────┘ └────────┘          │               │
│  │                 + optional Watermark          │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  Outputs: outputs/ outputs_ocr/ outputs_steps/                  │
│           outputs_redacted/ outputs_tokenized/ outputs_restored/ │
└──────────────────────────────────────────────────────────────────┘
```

### Component Interaction

```
CLI (main.py)  ──┐
                 ├──► mask_images()  ──► OCR ──► Detector ──► Blur/Redact/Tokenize
API (api.py)   ──┘
                              ▲
                        load_config()
                        (config.yaml / env)
```

---

## 3. Features

| # | Feature | Details |
|---|---------|---------|
| 1 | **OCR** | EasyOCR with configurable language packs, GPU toggle, canvas size, magnification ratio, and contrast adjustment |
| 2 | **Image Pre-processing** | Greyscale conversion, CLAHE contrast enhancement, NlMeansDenoising, sharpening, upscale/downscale to target width |
| 3 | **NER** | spaCy (`en_core_web_sm` default) or Hugging Face token-classification models |
| 4 | **Regex Detection** | Email, phone, invoice ID/ref, date, zip code, street address, and privacy-keyword patterns |
| 5 | **Zero-Shot Classification** | Optional secondary filter using a Hugging Face zero-shot model (`valhalla/distilbart-mnli-12-1`) to confirm sensitivity; combinable with NER/regex in `and` / `or` mode |
| 6 | **Gaussian Blur Masking** | Kernel size calculated proportionally to bounding-box dimensions; configurable padding, divisor, and minimum kernel |
| 7 | **Dummy-Text Redaction** | Two modes: random alphanumeric characters, or Faker-generated realistic substitutes matched to the detected field type |
| 8 | **Reversible Tokenisation** | Sensitive fields replaced with unique opaque tokens (`USER_XXXXXX`, `ORG_XXXXXX`, etc.); token ↔ original mapping persisted as JSON; restoring re-renders originals pixel-accurately |
| 9 | **Watermark** | Configurable diagonal text watermark stamped on every output image |
| 10 | **PDF Support** | Multi-page PDFs rendered at configurable DPI via Poppler/pdf2image; each page processed independently |
| 11 | **FastAPI UI + REST API** | Web form for browser-based use; JSON/file REST endpoints for integration |
| 12 | **Docker** | Single-file `Dockerfile`; `docker-compose.yml` for one-command deployment |
| 13 | **Step-by-Step Debug Outputs** | Optional per-stage image and JSON exports to trace every processing decision |
| 14 | **Configurable via YAML** | All knobs exposed in `config.yaml`; overridable via environment variable |

---

## 4. Technology Stack

| Layer | Library / Tool | Version | Purpose |
|-------|---------------|---------|---------|
| Language | Python | 3.10 + | Core runtime |
| OCR | EasyOCR | 1.7.1 | Text extraction from images |
| NLP / NER | spaCy | 3.7.6 | Named-entity recognition (default) |
| NLP / NER | Hugging Face Transformers | 4.44.2 | Alternative NER & zero-shot classifier |
| ML backend | PyTorch | 2.4.1 | Transformer model inference |
| Computer Vision | OpenCV | 4.10.0 | Image processing, blur, drawing |
| PDF Rendering | pdf2image + Pillow | 1.17.0 / 10.4.0 | PDF → image conversion |
| Data Generation | Faker | 25.8.0 | Realistic dummy values |
| Web Framework | FastAPI | 0.115.0 | REST API and HTML UI |
| ASGI Server | Uvicorn | 0.30.6 | Production-ready server |
| HTML Templates | Jinja2 | 3.1.4 | Server-side rendering |
| Configuration | PyYAML | 6.0.2 | YAML config loading |
| Container | Docker / docker-compose | — | Packaging and deployment |

---

## 5. Module Reference

### 5.1 Entry Points

#### `invoice_masker/main.py`

The primary orchestration module. Exposes the `mask_images()` function and the `main()` CLI entry point.

**Key functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `mask_images()` | `(images, output_dir, …, config, flags…) → List[str]` | Iterates pages, runs OCR → detection → output operations, returns list of output file paths |
| `build_detector()` | `(config, backend) → SensitiveDataDetector` | Instantiates the correct NER backend and optional classifier |
| `run_cli()` | `(args, config) → None` | Drives CLI masking workflow |
| `_run_restore()` | `(args, config) → None` | Restores a previously tokenised image |
| `_save_step_outputs()` | `(…) → None` | Writes per-stage debug files |
| `_load_token_map_for_image()` | `(input_path, token_dir) → List[dict]` | Locates and loads the token map JSON for a given image |

#### `invoice_masker/api.py`

FastAPI application. Wraps `mask_images()` and restoration logic behind HTTP endpoints and a Jinja2 HTML template.

---

### 5.2 OCR Engine

**File:** `invoice_masker/ocr/ocr_engine.py`

**Class:** `EasyOCREngine`

| Method | Description |
|--------|-------------|
| `__init__(config: OCRConfig)` | Loads (or reuses via LRU cache) an `easyocr.Reader` for the configured language set |
| `extract(image) → (results, scale)` | Runs OCR; if an OOM error occurs, automatically falls back to a downscaled image; returns bounding boxes with text and confidence scores, plus the effective scale factor |
| `_preprocess(image) → (image, scale)` | Optional pipeline: greyscale, denoise, CLAHE, sharpen, up/downscale |

The reader object is cached with `@lru_cache(maxsize=2)` keyed on `(languages, gpu)` so repeated calls do not reload model weights.

---

### 5.3 NER & Sensitive-Data Detection

**File:** `invoice_masker/ner/detector.py`

#### Class Hierarchy

```
BaseNERDetector
├── SpacyNERDetector        uses spaCy pipeline
└── HuggingFaceNERDetector  uses transformers token-classification pipeline
```

#### `SensitiveDataDetector`

Orchestrates NER, regex, and optional zero-shot classifier.

**Regex Patterns:**

| Pattern name | What it matches |
|-------------|-----------------|
| `EMAIL_REGEX` | Standard email addresses |
| `PHONE_REGEX` | Local and international phone numbers |
| `INVOICE_ID_REGEX` | `INV-NNNN` style identifiers |
| `INVOICE_REF_REGEX` | "Invoice No:", "Invoice #", etc. |
| `DATE_REGEX` | `DD/MM/YYYY`, `MM-DD-YY`, month-name formats |
| `ZIP_REGEX` | US-style 5-digit (and ZIP+4) postal codes |
| `ADDRESS_REGEX` | Numbered street addresses with common suffixes |
| `KEYWORD_REGEX` | Privacy-relevant labels: *Bill To, Ship To, Customer, Email*, etc. |

**NER Target Labels:** `PERSON`, `ORG`, `GPE` (mapped from Hugging Face's `PER`, `ORG`, `LOC`).

**Classification (optional):**

When `classification.enabled = true`, each OCR token is also scored by a zero-shot classification model against the candidate labels `["personal data", "financial data", "address or contact", "non-sensitive"]`. Sensitivity is determined by combining this result with NER/regex hits using `mode: "and"` (both must agree) or `mode: "or"` (either is sufficient).

**Key methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `detect_sensitive_regions(ocr_results)` | `(boxes, fields)` | Lightweight call that returns bounding boxes only |
| `detect_sensitive_details(ocr_results)` | `(boxes, fields, details)` | Full call returning per-token detail dicts used by redaction, tokenisation, and step outputs |

---

### 5.4 Blur (Masking)

**File:** `invoice_masker/utils/blur.py`

**Function:** `blur_regions(image, boxes, padding, blur_divisor, min_kernel) → np.ndarray`

For every sensitive bounding box:
1. Expands the box by `padding` pixels on all sides (clamped to image bounds).
2. Computes Gaussian kernel size proportionally: `kx = max(min_kernel, (box_width // blur_divisor) * 2 + 1)`.
3. Applies `cv2.GaussianBlur` to the region-of-interest in-place on a copy of the image.

---

### 5.5 Redaction

**File:** `invoice_masker/utils/redact.py`

**Function:** `redact_with_dummy_text(image, ocr_results, sensitive_boxes, …) → np.ndarray`

Two modes:

| Mode | `smart_faker` | Behaviour |
|------|--------------|-----------|
| Standard | `False` | Fills box with a random string of uppercase letters and digits, same length as the original |
| Smart Faker | `True` | Generates realistic replacement using the Python `Faker` library, type-aware (e.g. `faker.email()`, `faker.name()`, `faker.street_address()`) |

The field type is determined from the `details` dict produced by `SensitiveDataDetector.detect_sensitive_details()`.

---

### 5.6 Tokenizer

**File:** `invoice_masker/utils/tokenizer.py`

Implements **reversible pseudonymisation**.

**Token format:** `<PREFIX><6_HEX_CHARS>` (e.g. `USER_A3F21C`, `EMAIL_9B04D7`)

**Prefix assignment priority (highest → lowest):**
EMAIL → PHONE → ADDRESS → ZIP → DATE → INVOICE → PERSON → ORG → GPE → GEN

**Token database:** Stored as a JSON file (`outputs_tokenized/token_map.json`) mapping `original_text → { token, labels, regex_hits }`. The same original text always receives the same token within a run.

**Key functions:**

| Function | Description |
|----------|-------------|
| `tokenize_image(image, details, cfg)` | Renders tokens over sensitive boxes and persists the token DB |
| `restore_image_from_token_map(image, entries)` | Re-renders originals using per-image `_tokenized_map.json` (preferred path) |
| `restore_image_from_tokens_ocr(image, ocr_results, cfg)` | Re-renders originals by OCR-scanning a tokenised image and looking up tokens in the global DB |
| `restore_image_from_tokens(image, details, cfg)` | Re-renders originals using in-memory detail list |

---

### 5.7 Watermark

**File:** `invoice_masker/utils/watermark.py`

**Function:** `add_text_watermark(image, text, opacity, font_scale, …) → np.ndarray`

Draws a text label centred on the image using `cv2.putText`, then blends it over the original using `cv2.addWeighted` at the configured `opacity` (0 = invisible, 1 = opaque). Default text: `"CONFIDENTIAL - DEMO COPY"`.

---

### 5.8 PDF Handler

**File:** `invoice_masker/utils/pdf_handler.py`

**Function:** `load_images_from_path(path, pdf_config) → List[np.ndarray]`

- Accepts `.pdf`, `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`.
- PDFs are rendered to PIL images via `pdf2image.convert_from_path` at the configured DPI, then converted to BGR NumPy arrays for OpenCV compatibility.
- RGBA images are converted to BGR; greyscale images are converted to BGR.

---

### 5.9 Configuration

**File:** `invoice_masker/utils/config.py`

All settings are represented as frozen Python `dataclass` objects. The `load_config()` function:
1. Resolves the config path from the function argument, `INVOICE_MASKER_CONFIG` environment variable, or the default `config.yaml` sibling.
2. Deep-merges the YAML overrides on top of the dataclass defaults.
3. Returns a fully typed `AppConfig` object.

**Configuration classes:**

| Class | Key fields |
|-------|-----------|
| `OCRConfig` | languages, gpu, text_threshold, low_text, link_threshold, canvas_size, mag_ratio, contrast_ths, adjust_contrast, preprocess |
| `PreprocessConfig` | enabled, upscale_max_width, downscale_max_width, clahe_clip, denoise, sharpen |
| `NERConfig` | backend (spacy/huggingface), spacy_model, hf_model |
| `MaskingConfig` | padding, blur_divisor, min_kernel |
| `PDFConfig` | dpi, thread_count |
| `RedactionConfig` | smart_faker, charset, min_len |
| `TokenizationConfig` | output_dir, db_path, restore_output_dir, prefix_map |
| `ClassificationConfig` | enabled, model_name, candidate_labels, threshold, mode |
| `LoggingConfig` | level, file, console |
| `AppConfig` | All of the above + output directories + watermark settings |

---

### 5.10 Logger

**File:** `invoice_masker/utils/logger.py`

Sets up Python's `logging` module with:
- A rotating file handler writing to `logs/invoice_masker.log`.
- An optional `StreamHandler` for console output.
- Log level read from `LoggingConfig`.

---

## 6. Processing Pipeline

The following describes the end-to-end flow for a single invoice page:

```
Step 1 – Load
  load_images_from_path(input_path, pdf_config)
  → list of OpenCV BGR images

Step 2 – OCR
  EasyOCREngine.extract(image)
  → [(bbox, text, confidence), …]

Step 3 – Optional Image Pre-processing (inside extract)
  greyscale → denoise → CLAHE → sharpen → resize

Step 4 – Sensitive Data Detection
  SensitiveDataDetector.detect_sensitive_details(ocr_results)
  → sensitive_boxes, detected_field_labels, per_token_details

  For each OCR token:
    a. Regex matching (email, phone, date, address, …)
    b. NER (spaCy or HuggingFace)
    c. Optional zero-shot classifier
    d. Combine via AND / OR mode
    → sensitive: True / False

Step 5 – Output Operations (any combination):

  [A] Blur Masking
      blur_regions(image, sensitive_boxes, …)
      → masked_image

  [B] Redaction (--save-dummy)
      redact_with_dummy_text(image, ocr_results, sensitive_boxes, …)
      → redacted_image

  [C] Tokenisation (--save-tokenized)
      tokenize_image(image, details, token_cfg)
      → tokenized_image + token_map.json

Step 6 – Watermark (unless --no-watermark)
  add_text_watermark(image, text, opacity, …)

Step 7 – Write Outputs
  cv2.imwrite(output_path, result_image)

Step 8 – Optional: OCR boxes (--save-ocr-boxes)
  JSON + annotated PNG of all detected text boxes

Step 9 – Optional: Step outputs (--save-steps)
  step2_text/  → raw OCR JSON
  step3_ner_regex/  → full detail JSON
  step4_sensitive/ → annotated overlay (red = sensitive, green = safe)
  step5_masked/  → masked preview

Step 10 – Optional: Restore (--restore-tokenized)
  Locate token map → restore_image_from_token_map()
  or fallback: re-OCR tokenized image → restore_image_from_tokens_ocr()
```

---

## 7. API Reference

The FastAPI server is started with:
```bash
uvicorn api:app --reload        # development
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2  # production
```

The interactive Swagger documentation is available at `http://host:8000/docs`.

---

### `GET /`

Returns the HTML web interface (`templates/index.html`).

---

### `POST /mask`

Masks a single invoice file and returns the masked image.

**Request (multipart/form-data):**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | UploadFile | *required* | Invoice image (PNG/JPG) or PDF |
| `ner_backend` | string | `spacy` | NER backend: `spacy` or `huggingface` |
| `save_ocr_boxes` | bool | `true` | Save OCR bounding box artefacts |
| `save_steps` | bool | `false` | Save step-by-step debug artefacts |
| `save_dummy` | bool | `false` | Save redacted-dummy output |
| `smart_faker` | bool | `false` | Use Faker for realistic dummy data |
| `save_tokenized` | bool | `false` | Save tokenised output + token map |
| `apply_watermark` | bool | `true` | Stamp watermark on output |

**Response:**

- `200 OK` — `image/png` file (first page if PDF)
- `X-Page-Count` response header — total number of pages processed
- `400 Bad Request` — missing filename or unreadable file
- `500 Internal Server Error` — processing failure

---

### `POST /restore`

Restores a previously tokenised image to its original text.

**Request (multipart/form-data):**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | UploadFile | Tokenised invoice image |

**Response:**

- `200 OK` — `image/png` restored image
- `X-Page-Count` — total pages
- `400 Bad Request` — missing or unreadable file

---

### `POST /upload`

HTML form submission endpoint. Processes an invoice and returns an HTML page with links to all generated artefacts.

**Request (multipart/form-data):** Same parameters as `POST /mask`.

**Response:** HTML page with output links.

---

### `POST /restore_upload`

HTML form submission for restoration. Returns an HTML page with links to restored artefacts.

---

### Static File Mounts

| URL prefix | Serves directory |
|------------|-----------------|
| `/outputs` | `outputs/` |
| `/outputs_ocr` | `outputs_ocr/` |
| `/outputs_steps` | `outputs_steps/` |
| `/outputs_redacted` | `outputs_redacted/` |
| `/outputs_tokenized` | `outputs_tokenized/` |
| `/outputs_restored` | `outputs_restored/` |

---

## 8. CLI Reference

Run from within the `invoice_masker/` directory:

```bash
cd invoice_masker
python main.py --input <path>  [options]
```

### Required argument

| Argument | Description |
|----------|-------------|
| `--input PATH` | Path to the invoice image (PNG/JPG/TIFF/BMP) or PDF |

### Optional arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir DIR` | `config.outputs_dir` | Directory for masked output images |
| `--ner-backend {spacy,huggingface}` | `config.ner.backend` | NER backend selection |
| `--config PATH` | `config.yaml` | Override config file path |
| `--save-ocr-boxes` | off | Save OCR bounding box JSON + PNG in `outputs_ocr/` |
| `--save-steps` | off | Save per-step debug files in `outputs_steps/` |
| `--save-dummy` | off | Save redacted output in `outputs_redacted/` |
| `--smart-faker` | off | Use Faker for realistic dummy replacement |
| `--save-tokenized` | off | Save tokenised output + map in `outputs_tokenized/` |
| `--no-watermark` | off | Disable watermark on all outputs |
| `--restore-tokenized` | off | After masking, also restore the tokenised image |

### Examples

```bash
# Basic masking
python main.py --input invoices/invoice.png

# Full pipeline with all optional outputs
python main.py --input invoices/invoice.pdf \
    --save-ocr-boxes --save-steps --save-dummy --save-tokenized

# Smart redaction with Faker
python main.py --input invoices/invoice.png --save-dummy --smart-faker

# Restore a previously tokenised image
python main.py --input outputs_tokenized/invoice_tokenized.png \
    --restore-tokenized

# Use Hugging Face NER instead of spaCy
python main.py --input invoices/invoice.png --ner-backend huggingface
```

---

## 9. Configuration Reference

The configuration file is located at `invoice_masker/config.yaml`. An alternative path can be supplied via the `INVOICE_MASKER_CONFIG` environment variable.

```yaml
# Output directory paths (relative to invoice_masker/)
outputs_dir: outputs
outputs_ocr_dir: outputs_ocr
outputs_steps_dir: outputs_steps
outputs_redacted_dir: outputs_redacted

ocr:
  languages: ["en"]           # EasyOCR language codes
  gpu: false                  # Enable GPU acceleration
  text_threshold: 0.7         # Minimum text detection confidence
  low_text: 0.4               # Low-boundary text score
  link_threshold: 0.4         # Link confidence threshold
  canvas_size: 2560           # Maximum image dimension for OCR canvas
  mag_ratio: 1.5              # Image magnification ratio
  contrast_ths: 0.1           # Contrast threshold for adjustment
  adjust_contrast: 0.5        # Contrast adjustment strength
  preprocess:
    enabled: true
    upscale_max_width: 1800   # Upscale small images to this width
    downscale_max_width: 2000 # Downscale large images to this width
    clahe_clip: 2.0           # CLAHE clip limit (0 = disabled)
    denoise: true             # Apply NlMeansDenoising
    sharpen: false            # Apply sharpening kernel

ner:
  backend: spacy              # "spacy" or "huggingface"
  spacy_model: en_core_web_sm
  hf_model: dslim/bert-base-NER

masking:
  padding: 2                  # Extra pixels added around each box
  blur_divisor: 7             # Kernel = (box_size // divisor) * 2 + 1
  min_kernel: 3               # Minimum Gaussian kernel size

pdf:
  dpi: 250                    # Render DPI for PDF pages
  thread_count: 2             # pdf2image thread count

redaction:
  smart_faker: false          # Use Faker for realistic substitutes
  charset: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
  min_len: 4                  # Minimum length of dummy text

tokenization:
  output_dir: outputs_tokenized
  db_path: outputs_tokenized/token_map.json
  restore_output_dir: outputs_restored
  prefix_map:
    PERSON: "USER_"
    ORG: "ORG_"
    GPE: "LOC_"
    EMAIL: "EMAIL_"
    PHONE: "PHONE_"
    ADDRESS: "ADDR_"
    ZIP: "ZIP_"
    DATE: "DATE_"
    INVOICE: "INV_"
    GEN: "GEN_"

classification:
  enabled: true
  model_name: "valhalla/distilbart-mnli-12-1"
  candidate_labels:
    - "personal data"
    - "financial data"
    - "address or contact"
    - "non-sensitive"
  threshold: 0.6
  mode: "and"                 # "and" = NER/regex AND classifier; "or" = either

watermark:
  enabled: true
  text: "CONFIDENTIAL - DEMO COPY"
  opacity: 0.25               # 0.0 (invisible) – 1.0 (opaque)
  font_scale: 1.2

logging:
  level: INFO
  file: logs/invoice_masker.log
  console: true
```

---

## 10. Output Artefacts

| Directory | Filename pattern | Content |
|-----------|-----------------|---------|
| `outputs/` | `<name>_masked.png` | Primary blurred/masked image |
| `outputs_ocr/` | `<name>_ocr_boxes.json` | All OCR detections (text, confidence, bbox) |
| `outputs_ocr/` | `<name>_ocr_boxes.png` | Image annotated with all OCR boxes |
| `outputs_steps/step2_text/` | `<name>.json` | Raw OCR text output |
| `outputs_steps/step3_ner_regex/` | `<name>.json` | Per-token NER + regex + classifier details |
| `outputs_steps/step4_sensitive/` | `<name>.json` | Sensitivity flags per token |
| `outputs_steps/step4_sensitive/` | `<name>.png` | Image with red (sensitive) / green (safe) boxes |
| `outputs_steps/step5_masked/` | `<name>.png` | Masked preview |
| `outputs_redacted/` | `<name>_redacted.png` | Dummy-text redacted image |
| `outputs_tokenized/` | `<name>_tokenized.png` | Image with opaque tokens |
| `outputs_tokenized/` | `<name>_tokenized_map.json` | Token ↔ original mapping for this image |
| `outputs_tokenized/` | `token_map.json` | Global persistent token database |
| `outputs_restored/` | `<name>_restored.png` | Image with original text re-rendered |

Multi-page PDFs receive a `_page_N` suffix before the operation qualifier.

---

## 11. Deployment

### Local (Python venv)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

# 2. Install dependencies
pip install -r invoice_masker/requirements.txt

# 3. Download spaCy model (default NER backend)
python -m spacy download en_core_web_sm

# 4. (Optional) Set a custom config path
export INVOICE_MASKER_CONFIG=/path/to/my_config.yaml

# 5. Start API server
cd invoice_masker
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 2
```

### Docker

```bash
# Build image
docker build -t invoice-masker .

# Run container
docker run --rm -p 8000:8000 invoice-masker
```

### Docker Compose

```bash
docker-compose up --build
```

Access the web UI at `http://localhost:8000`.

### Environment Variables

| Variable | Description |
|----------|-------------|
| `INVOICE_MASKER_CONFIG` | Absolute path to a custom `config.yaml` file |

---

## 12. Security & Privacy Design

| Concern | Implementation |
|---------|---------------|
| **Data minimisation** | All uploaded files are written to a temporary directory (`tempfile.mkdtemp`) and deleted by a FastAPI `BackgroundTask` after the response is sent |
| **No external data transmission** | All ML inference runs locally; no invoice data leaves the server |
| **Reversibility control** | Tokenisation is the only reversible operation; the token map is stored server-side and is never returned to the caller unless explicitly requested |
| **Watermarking** | Every output image carries a configurable watermark to prevent accidental misuse of masked copies |
| **Unique filenames** | The API appends an 8-character random hex suffix to each uploaded filename to prevent collisions and path-traversal risks |
| **No secrets in config** | The config file and environment variable accept paths only; no credentials are stored |
| **Input validation** | Unsupported file types return `HTTP 400`; empty files are rejected before processing |

---

## 13. Testing

Tests are located in `tests/` and use **pytest**.

```bash
pytest
```

### Test suite

| Test | Description |
|------|-------------|
| `test_imports` | Verifies that the main application modules can be imported without error |
| `test_config_loads` | Verifies that `load_config()` returns a valid `AppConfig` object |

Run with:
```bash
pytest --tb=short -v
```

Configuration is in `pytest.ini`.

---

## 14. Directory Structure

```
invoice-privacy-and-Blur-tool/
├── .env.example                  # Example environment variable file
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── README.md
├── DOCUMENTATION.md              # ← This document
│
├── docs/
│   └── preview.gif               # Animated demo
│
├── tests/
│   ├── __init__.py
│   └── test_smoke.py
│
└── invoice_masker/               # Main application package
    ├── __init__.py
    ├── main.py                   # CLI entry point & core orchestration
    ├── api.py                    # FastAPI application
    ├── config.yaml               # Default configuration
    ├── requirements.txt
    │
    ├── ner/
    │   └── detector.py           # NER backends & SensitiveDataDetector
    │
    ├── ocr/
    │   └── ocr_engine.py         # EasyOCR wrapper
    │
    ├── utils/
    │   ├── blur.py               # Gaussian blur masking
    │   ├── config.py             # Dataclass config + YAML loader
    │   ├── logger.py             # Logging setup
    │   ├── pdf_handler.py        # PDF/image loading
    │   ├── redact.py             # Dummy-text redaction
    │   ├── tokenizer.py          # Reversible tokenisation
    │   └── watermark.py          # Text watermark overlay
    │
    ├── templates/
    │   └── index.html            # Jinja2 web UI template
    │
    ├── outputs/                  # Masked images
    ├── outputs_ocr/              # OCR box artefacts
    ├── outputs_steps/            # Step-debug artefacts
    ├── outputs_redacted/         # Dummy-text redacted images
    ├── outputs_tokenized/        # Tokenised images + maps
    └── outputs_restored/         # Restored images
```

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **OCR** | Optical Character Recognition — extracting text from images |
| **NER** | Named-Entity Recognition — classifying text spans as PERSON, ORG, date, etc. |
| **Bounding Box (bbox)** | Rectangle `[x_min, y_min, x_max, y_max]` enclosing a detected text region |
| **Masking / Blur** | Applying Gaussian blur to a bounding box to obscure its content visually |
| **Redaction** | Replacing sensitive text with random or plausible dummy text |
| **Tokenisation** | Replacing sensitive text with a unique opaque token that can be reversed |
| **Token Map** | JSON database mapping each token back to the original text |
| **spaCy** | Open-source Python NLP library; used here for English NER |
| **Hugging Face** | ML model hub and library ecosystem; used for alternative NER and zero-shot classification |
| **Zero-Shot Classification** | Classifying text into arbitrary categories without task-specific training data |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalisation — local contrast enhancement |
| **Faker** | Python library for generating realistic fake personal data |
| **PII** | Personally Identifiable Information (names, emails, addresses, phone numbers, etc.) |
| **FastAPI** | Modern Python web framework for building APIs with automatic OpenAPI documentation |
| **Uvicorn** | ASGI server used to run FastAPI applications |
| **Poppler** | PDF rendering library required by `pdf2image` |
