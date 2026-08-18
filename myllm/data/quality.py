"""Deterministic, dependency-free quality checks for Dhruva training text."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
import unicodedata
from typing import Any, Iterable


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{19,}", re.DOTALL)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_HTML_RE = re.compile(r"</?[a-zA-Z][^>]{0,200}>")


LANGUAGE_SCRIPT = {
    "arabic": "Arabic",
    "bengali": "Bengali",
    "bangla": "Bengali",
    "bn": "Bengali",
    "chinese": "Han",
    "zh": "Han",
    "english": "Latin",
    "en": "Latin",
    "hindi": "Devanagari",
    "hi": "Devanagari",
    "japanese": "Japanese",
    "ja": "Japanese",
    "korean": "Hangul",
    "ko": "Hangul",
    "russian": "Cyrillic",
    "ru": "Cyrillic",
    "sanskrit": "Devanagari",
    "sa": "Devanagari",
    "tamil": "Tamil",
    "ta": "Tamil",
    "telugu": "Telugu",
    "te": "Telugu",
    "urdu": "Arabic",
    "ur": "Arabic",
}


@dataclass(frozen=True)
class QualityPolicy:
    min_chars: int = 160
    max_chars: int = 200_000
    min_visible_ratio: float = 0.85
    min_alnum_ratio: float = 0.20
    max_html_ratio: float = 0.08
    max_control_ratio: float = 0.001
    min_word_diversity: float = 0.12
    require_metadata: bool = True
    required_metadata: tuple[str, ...] = (
        "source", "source_revision", "language", "domain", "license",
        "document_id", "preprocessing_version",
    )


@dataclass
class QualityResult:
    accepted: bool
    normalized_text: str
    content_hash: str
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dominant_script: str = "Unknown"


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = [_WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _script_name(char: str) -> str | None:
    code = ord(char)
    if 0x0041 <= code <= 0x024F:
        return "Latin"
    if 0x0400 <= code <= 0x052F:
        return "Cyrillic"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F:
        return "Arabic"
    if 0x0900 <= code <= 0x097F:
        return "Devanagari"
    if 0x0980 <= code <= 0x09FF:
        return "Bengali"
    if 0x0B80 <= code <= 0x0BFF:
        return "Tamil"
    if 0x0C00 <= code <= 0x0C7F:
        return "Telugu"
    if 0x3040 <= code <= 0x30FF:
        return "Japanese"
    if 0x4E00 <= code <= 0x9FFF:
        return "Han"
    if 0xAC00 <= code <= 0xD7AF:
        return "Hangul"
    return None


def script_distribution(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_name(char)
        if script:
            counts[script] = counts.get(script, 0) + 1
    return counts


def dominant_script(text: str) -> str:
    counts = script_distribution(text)
    if not counts:
        return "Unknown"
    if counts.get("Japanese", 0) > 0:
        return "Japanese"
    return max(counts, key=counts.get)


def _language_script_warning(language: Any, detected: str) -> str | None:
    key = str(language or "").strip().lower()
    expected = LANGUAGE_SCRIPT.get(key)
    if not expected or detected == "Unknown":
        return None
    if expected == "Japanese" and detected in {"Japanese", "Han"}:
        return None
    if expected != detected:
        return f"language_script_mismatch:{expected}!={detected}"
    return None


def analyze_document(record: dict[str, Any], policy: QualityPolicy | None = None) -> QualityResult:
    policy = policy or QualityPolicy()
    raw_text = record.get("text")
    if not isinstance(raw_text, str):
        return QualityResult(False, "", content_hash(""), ["missing_text"])

    text = normalize_text(raw_text)
    reasons: list[str] = []
    warnings: list[str] = []

    if len(text) < policy.min_chars:
        reasons.append("too_short")
    if len(text) > policy.max_chars:
        reasons.append("too_long")
    if "\ufffd" in text:
        reasons.append("unicode_replacement_character")
    if _REPEATED_CHAR_RE.search(text):
        reasons.append("repeated_character_run")

    length = max(1, len(text))
    visible = sum(not char.isspace() and not unicodedata.category(char).startswith("C") for char in text)
    alnum = sum(char.isalnum() for char in text)
    controls = sum(unicodedata.category(char).startswith("C") and char not in "\n\t" for char in text)
    html_chars = sum(len(match.group(0)) for match in _HTML_RE.finditer(text))

    if visible / length < policy.min_visible_ratio:
        reasons.append("low_visible_character_ratio")
    if alnum / length < policy.min_alnum_ratio:
        reasons.append("low_alphanumeric_ratio")
    if controls / length > policy.max_control_ratio:
        reasons.append("excess_control_characters")
    if html_chars / length > policy.max_html_ratio:
        reasons.append("html_boilerplate")

    words = [word.casefold() for word in _WORD_RE.findall(text)]
    if len(words) >= 40 and len(set(words)) / len(words) < policy.min_word_diversity:
        reasons.append("low_word_diversity")

    if policy.require_metadata:
        for key in policy.required_metadata:
            value = record.get(key)
            if value is None or not str(value).strip():
                reasons.append(f"missing_metadata:{key}")
            elif key in {"source_revision", "document_id", "preprocessing_version"} and str(value).strip().lower() in {"unknown", "latest", "unversioned"}:
                reasons.append(f"unfrozen_metadata:{key}")

    detected_script = dominant_script(text)
    script_warning = _language_script_warning(record.get("language"), detected_script)
    if script_warning:
        reasons.append(script_warning)

    return QualityResult(
        accepted=not reasons,
        normalized_text=text,
        content_hash=content_hash(text.casefold()),
        reasons=reasons,
        warnings=warnings,
        dominant_script=detected_script,
    )


def flatten_text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value.strip():
            yield normalize_text(value)
    elif isinstance(value, dict):
        for child in value.values():
            yield from flatten_text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from flatten_text_values(child)
