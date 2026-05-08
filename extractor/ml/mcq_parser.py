"""
MCQ Parser
==========
Takes a flat list of Block objects (the combined content of one question group)
and splits it into:
  - question_number  (int | None)
  - marks            (float | None)
  - question_type    (QuestionType)
  - content          List[Block]   — everything before the first option marker
  - options          Dict[str, List[Block]]  — "a", "b", "c", "d" → Block list

Option marker patterns recognised
──────────────────────────────────
  (A) / (a) / A. / a) / A) / (1) / 1. / 1) / (i) / (ii) / (iii) / (iv)

Numeric markers 1-4 are mapped to a-d; roman i-iv are mapped to a-d.

Question type heuristics (checked in order)
──────────────────────────────────────────
  1. Options present ≥ 3          → mcq_single (or mcq_multi if multi-correct clue found)
  2. "Assertion" / "Reason"       → assertion_reason
  3. "integer" / "numerical"      → integer
  4. Otherwise                    → subjective
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from schema import Block, LatexBlock, QuestionType, TextBlock


# ── Compiled patterns ─────────────────────────────────────────────────────────

# Question-number prefix at the very start of a TextBlock value
# Captures the integer number as group 1
_Q_PREFIX = re.compile(
    r"^\s*[\(\[]?\s*Q?\.?\s*(\d+)\s*[.\)\]:\s]\s*",
    re.IGNORECASE,
)

# Option marker — must appear at the very start of a stripped string
# Group 1 captures the option letter/digit/roman
_OPTION_RE = re.compile(
    r"^[\(\[]?\s*"
    r"([AaBbCcDd]|[1-4]|[iI][iI][iI][iI]|[iI][iI][iI]|[iI][iI]|[iI][vV]|[iI])"
    r"\s*[\)\].\s]\s*",
)

# Marks clues: "[2 Marks]", "(3 marks)", "2M", "Marks : 4"
_MARKS_RE  = re.compile(r"[\[\(]?\s*(\d+(?:\.\d+)?)\s*[Mm]ark", re.IGNORECASE)

# Question-type classifiers (applied to full concatenated text of content)
_INTEGER_RE   = re.compile(
    r"\b(integer|numerical|value of|answer is (an? )?integer|find the value)\b",
    re.IGNORECASE,
)
_ASSERTION_RE = re.compile(
    r"\b(assertion|reason|statement[- ]?(i|ii|1|2)|both.*statement)\b",
    re.IGNORECASE,
)
_MULTI_RE     = re.compile(
    r"\b(one or more|correct option[s]?|may be correct|all.*correct|more than one)\b",
    re.IGNORECASE,
)

# Roman → letter mapping
_ROMAN_MAP = {"i": "a", "ii": "b", "iii": "c", "iv": "d"}

# Digit → letter mapping
_DIGIT_MAP = {"1": "a", "2": "b", "3": "c", "4": "d"}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_question(
    blocks: List[Block],
) -> Tuple[Optional[int], Optional[float], QuestionType, List[Block], Dict[str, List[Block]]]:
    """
    Parse the flat block list for a single question.

    Returns
    -------
    question_number, marks, question_type, content_blocks, options_dict
    """
    if not blocks:
        return None, None, QuestionType.UNKNOWN, [], {}

    blocks = list(blocks)  # work on a copy

    # ── Strip question-number prefix from the leading TextBlock ───────────────
    q_number: Optional[int] = None
    first_text_idx = next(
        (i for i, b in enumerate(blocks) if isinstance(b, TextBlock)), None
    )
    if first_text_idx is not None:
        tb = blocks[first_text_idx]
        m  = _Q_PREFIX.match(tb.value)
        if m:
            q_number  = int(m.group(1))
            remainder = tb.value[m.end():].strip()
            if remainder:
                blocks[first_text_idx] = TextBlock(value=remainder)
            else:
                blocks.pop(first_text_idx)

    # ── Extract marks ─────────────────────────────────────────────────────────
    marks: Optional[float] = None
    full_text = " ".join(b.value for b in blocks if isinstance(b, TextBlock))
    m_marks = _MARKS_RE.search(full_text)
    if m_marks:
        try:
            marks = float(m_marks.group(1))
        except ValueError:
            pass

    # ── Split content vs options ──────────────────────────────────────────────
    content: List[Block]              = []
    options: Dict[str, List[Block]]   = {}
    current_option: Optional[str]     = None

    for block in blocks:
        if isinstance(block, TextBlock):
            stripped = block.value.strip()
            opt_m    = _OPTION_RE.match(stripped)

            if opt_m:
                raw     = opt_m.group(1).lower()
                letter  = _ROMAN_MAP.get(raw, _DIGIT_MAP.get(raw, raw))
                rest    = stripped[opt_m.end():].strip()
                current_option = letter
                if letter not in options:
                    options[letter] = []
                if rest:
                    options[letter].append(TextBlock(value=rest))
                continue  # the marker line itself is consumed

        # Non-marker block (TextBlock with non-marker text, or LatexBlock etc.)
        if current_option is None:
            content.append(block)
        else:
            if current_option not in options:
                options[current_option] = []
            options[current_option].append(block)

    # ── Classify question type ────────────────────────────────────────────────
    q_type = _classify(full_text, options)

    return q_number, marks, q_type, content, options


# ── Internal helpers ──────────────────────────────────────────────────────────

def _classify(full_text: str, options: dict) -> QuestionType:
    if options and len(options) >= 3:
        if _MULTI_RE.search(full_text):
            return QuestionType.MCQ_MULTI
        return QuestionType.MCQ_SINGLE
    if _ASSERTION_RE.search(full_text):
        return QuestionType.ASSERTION_REASON
    if _INTEGER_RE.search(full_text):
        return QuestionType.INTEGER
    if options:
        # Fewer than 3 options but options exist — still probably MCQ
        return QuestionType.MCQ_SINGLE
    return QuestionType.SUBJECTIVE
