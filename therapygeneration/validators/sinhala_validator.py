# therapygeneration/validators/sinhala_validator.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import unicodedata
import re


# Sinhala Unicode block: U+0D80–U+0DFF
# Allow ZWJ/ZWNJ optionally for safety with some keyboards.
_ALLOWED_CODEPOINT_RANGES: List[Tuple[int, int]] = [
    (0x0D80, 0x0DFF),  # Sinhala block
]
_ALLOWED_CODEPOINTS: Set[int] = {0x200C, 0x200D}  # ZWNJ, ZWJ


def normalize_si(text: str) -> str:
    """Normalize Sinhala text deterministically for comparisons."""
    if text is None:
        return ""
    t = text.strip()
    # Normalize Unicode combining marks etc.
    return unicodedata.normalize("NFC", t)


def is_sinhala_only(word: str) -> bool:
    """Return True if the string contains only Sinhala block characters (plus ZWJ/ZWNJ)."""
    w = normalize_si(word)
    if not w:
        return False

    for ch in w:
        cp = ord(ch)
        if cp in _ALLOWED_CODEPOINTS:
            continue
        ok = any(start <= cp <= end for start, end in _ALLOWED_CODEPOINT_RANGES)
        if not ok:
            return False
    return True


def matches_mode(word: str, letter: str, mode: str) -> bool:
    """Deterministically check phonological position mode using string matching."""
    w = normalize_si(word)
    l = normalize_si(letter)
    if not w or not l:
        return False

    if mode == "starts_with":
        return w.startswith(l)
    if mode == "contains":
        return l in w
    if mode == "ends_with":
        return w.endswith(l)

    # Unknown mode
    return False


def length_ok(word: str, max_len: Optional[int]) -> bool:
    """Simple length rule (codepoint length). Good enough for V1."""
    if max_len is None:
        return True
    w = normalize_si(word)
    return 1 <= len(w) <= max_len


@dataclass(frozen=True)
class ValidationResult:
    word: str
    normalized: str
    valid: bool
    reasons: List[str]


def validate_word_candidate(
    word: str,
    *,
    letter: str,
    mode: str,
    max_len: Optional[int] = None,
    existing_words_normalized: Optional[Set[str]] = None,
    seen_normalized: Optional[Set[str]] = None,
) -> ValidationResult:
    """
    Validate a candidate word deterministically.
    existing_words_normalized: normalized words already in DB (optional)
    seen_normalized: normalized words already seen in current candidate list (optional)
    """
    reasons: List[str] = []
    w_norm = normalize_si(word)

    if not w_norm:
        reasons.append("empty")

    if w_norm and not is_sinhala_only(w_norm):
        reasons.append("non_sinhala_characters")

    if w_norm and not matches_mode(w_norm, letter, mode):
        reasons.append("does_not_match_position_mode")

    if w_norm and not length_ok(w_norm, max_len):
        reasons.append("too_long_for_level")

    if existing_words_normalized is not None and w_norm and w_norm in existing_words_normalized:
        reasons.append("already_exists_in_db")

    if seen_normalized is not None and w_norm and w_norm in seen_normalized:
        reasons.append("duplicate_in_suggestions")

    valid = len(reasons) == 0
    return ValidationResult(word=word, normalized=w_norm, valid=valid, reasons=reasons)


def validate_candidate_list(
    candidates: List[str],
    *,
    letter: str,
    mode: str,
    max_len: Optional[int] = None,
    existing_words_normalized: Optional[Set[str]] = None,
) -> List[ValidationResult]:
    """
    Validate a list of candidate words. Flags duplicates within the candidate list.
    """
    seen: Set[str] = set()
    results: List[ValidationResult] = []
    for w in candidates:
        r = validate_word_candidate(
            w,
            letter=letter,
            mode=mode,
            max_len=max_len,
            existing_words_normalized=existing_words_normalized,
            seen_normalized=seen,
        )
        results.append(r)
        if r.normalized:
            seen.add(r.normalized)
    return results
