import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Sequence, Tuple

import spacy
from transformers import pipeline


class TextClassifier:
    def __init__(
        self,
        model_name: str,
        candidate_labels: list[str],
        threshold: float = 0.6,
    ) -> None:
        self.model_name = model_name
        self.candidate_labels = candidate_labels
        self.threshold = threshold
        self.pipe = pipeline("zero-shot-classification", model=model_name)

    def is_sensitive(self, text: str) -> tuple[bool, dict]:
        result = self.pipe(text, self.candidate_labels, multi_label=True)
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        scored = dict(zip(labels, scores))
        sensitive_score = max(
            scored.get("personal data", 0.0),
            scored.get("financial data", 0.0),
            scored.get("address or contact", 0.0),
        )
        return sensitive_score >= self.threshold, scored


class NERBackend(str, Enum):
    SPACY = "spacy"
    HUGGINGFACE = "huggingface"


@dataclass(frozen=True)
class Entity:
    text: str
    label: str


class BaseNERDetector:
    def detect(self, text: str) -> List[Entity]:
        raise NotImplementedError


class SpacyNERDetector(BaseNERDetector):
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.nlp = spacy.load(model_name)

    def detect(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        entities: List[Entity] = []
        for ent in doc.ents:
            entities.append(Entity(text=ent.text, label=ent.label_))
        return entities


class HuggingFaceNERDetector(BaseNERDetector):
    def __init__(self, model_name: str = "dslim/bert-base-NER") -> None:
        self.pipe = pipeline(
            "ner",
            model=model_name,
            aggregation_strategy="simple",
        )

    def detect(self, text: str) -> List[Entity]:
        entities: List[Entity] = []
        for ent in self.pipe(text):
            label = ent.get("entity_group") or ent.get("entity")
            if label:
                entities.append(Entity(text=ent["word"], label=label))
        return entities


class SensitiveDataDetector:
    TARGET_LABELS = {"PERSON", "ORG", "GPE"}
    HF_LABEL_MAP = {"PER": "PERSON", "ORG": "ORG", "LOC": "GPE"}

    EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    PHONE_REGEX = re.compile(
        r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"
    )
    INVOICE_ID_REGEX = re.compile(r"\bINV-\d+\b", re.IGNORECASE)
    INVOICE_REF_REGEX = re.compile(
        r"\bInvoice\s*(?:No|#|Number)?\s*[:#]?\s*[A-Za-z0-9-]+\b",
        re.IGNORECASE,
    )
    DATE_REGEX = re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b",
        re.IGNORECASE,
    )
    ZIP_REGEX = re.compile(r"\b\d{5}(?:-\d{4})?\b")
    ADDRESS_REGEX = re.compile(
        r"\b\d{1,5}\s+[A-Za-z0-9.\-']+(?:\s+[A-Za-z0-9.\-']+){0,4}\s+"
        r"(?:St|Street|Rd|Road|Ave|Avenue|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place|Pkwy|Parkway)\b",
        re.IGNORECASE,
    )
    KEYWORD_REGEX = re.compile(
        r"\b(?:Bill\s*To|Ship\s*To|Customer|Receiver|Importer|Exporter|Consignee|"
        r"Contact\s*Person|Shipper|Buyer|Seller|Company|Address|Phone|Email)\b",
        re.IGNORECASE,
    )

    def __init__(
        self,
        ner_detector: BaseNERDetector,
        classifier: TextClassifier | None = None,
        classifier_mode: str = "and",
    ) -> None:
        self.ner_detector = ner_detector
        self._cache: dict[str, Tuple[List[Entity], List[str]]] = {}
        self.classifier = classifier
        self.classifier_mode = classifier_mode

    def _matches_regex(self, text: str) -> List[str]:
        matches = []
        if self.EMAIL_REGEX.search(text):
            matches.append("EMAIL")
        if self.PHONE_REGEX.search(text):
            matches.append("PHONE")
        if self.INVOICE_ID_REGEX.search(text):
            matches.append("INVOICE_ID")
        if self.INVOICE_REF_REGEX.search(text):
            matches.append("INVOICE_REF")
        if self.DATE_REGEX.search(text):
            matches.append("DATE")
        if self.ADDRESS_REGEX.search(text):
            matches.append("ADDRESS")
        if self.ZIP_REGEX.search(text):
            matches.append("ZIP")
        if self.KEYWORD_REGEX.search(text):
            matches.append("KEYWORD")
        return matches

    def detect_sensitive_regions(
        self,
        ocr_results: Sequence[Tuple[Iterable[Iterable[int]], str, float]],
    ) -> Tuple[List[List[int]], List[str]]:
        boxes, detected_fields, _details = self.detect_sensitive_details(ocr_results)
        return boxes, detected_fields

    def detect_sensitive_details(
        self,
        ocr_results: Sequence[Tuple[Iterable[Iterable[int]], str, float]],
    ) -> Tuple[List[List[int]], List[str], List[dict]]:
        boxes: List[List[int]] = []
        detected_fields: List[str] = []
        details: List[dict] = []

        for bbox, text, conf in ocr_results:
            text = text.strip()
            if not text:
                continue

            if text in self._cache:
                ner_entities, regex_hits = self._cache[text]
            else:
                try:
                    ner_entities = self.ner_detector.detect(text)
                except Exception:
                    ner_entities = []
                regex_hits = self._matches_regex(text)
                self._cache[text] = (ner_entities, regex_hits)

            ner_labels = {self._normalize_label(ent.label) for ent in ner_entities}
            ner_labels.discard(None)

            basic_sensitive = bool(
                ner_labels.intersection(self.TARGET_LABELS) or regex_hits
            )
            classifier_sensitive = False
            classifier_scores = {}
            if self.classifier:
                try:
                    classifier_sensitive, classifier_scores = self.classifier.is_sensitive(
                        text
                    )
                except Exception:
                    classifier_sensitive = False

            if self.classifier:
                if self.classifier_mode == "and":
                    is_sensitive = basic_sensitive and classifier_sensitive
                else:
                    is_sensitive = basic_sensitive or classifier_sensitive
            else:
                is_sensitive = basic_sensitive
            if is_sensitive:
                boxes.append(self._bbox_to_rect(bbox))
                detected_fields.extend(regex_hits)
                detected_fields.extend(
                    label for label in ner_labels if label in self.TARGET_LABELS
                )

            details.append(
                {
                    "text": text,
                    "confidence": float(conf),
                    "bbox": self._bbox_to_rect(bbox),
                    "ner_labels": sorted(
                        [label for label in ner_labels if label in self.TARGET_LABELS]
                    ),
                    "regex_hits": sorted(regex_hits),
                    "classifier_scores": classifier_scores,
                    "sensitive": is_sensitive,
                }
            )

        return boxes, list(set(detected_fields)), details

    @staticmethod
    def _bbox_to_rect(bbox: Iterable[Iterable[int]]) -> List[int]:
        xs = [int(point[0]) for point in bbox]
        ys = [int(point[1]) for point in bbox]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)
        return [x_min, y_min, x_max, y_max]

    def _normalize_label(self, label: str | None) -> str | None:
        if label is None:
            return None
        if label in self.TARGET_LABELS:
            return label
        return self.HF_LABEL_MAP.get(label)
