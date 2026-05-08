"""
Question Segmenter
==================
Groups an ordered list of (LayoutBlock, List[Block]) pairs into
per-question clusters based on:

  1. Question-number detection — a TextBlock whose text starts with a
     recognisable question marker (e.g. "Q.1", "1.", "(1)", "1)").

  2. Large vertical gap — if consecutive blocks are widely separated on
     the page, a new group may start even without an explicit number.

Returns
-------
ungrouped : List[Block]
    All blocks that appear *before* the first detected question number.
    Typically page headers, instructions, or section titles.

groups : List[QuestionGroup]
    One QuestionGroup per detected question, each containing the
    ordered (LayoutBlock, List[Block]) pairs that belong to it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from layout import LayoutBlock
from schema import Block, TextBlock


# ── Question-start pattern ────────────────────────────────────────────────────
# Matches: "1.", "1)", "(1)", "Q.1", "Q1.", "Q1)", "Q. 1." etc.
# Captures the integer question number as group 1.
_Q_START = re.compile(
    r"^\s*[\(\[]?\s*Q?\.?\s*(\d+)\s*[.\)\]:\s]",
    re.IGNORECASE,
)

# How many leading blocks to inspect when looking for a question marker
_LOOKAHEAD = 3

# Pixel gap threshold between the bottom of one block and the top of the next
# that (on its own, without a number marker) triggers a new question group.
# Set conservatively high — number detection is the primary signal.
_Y_GAP_THRESHOLD = 80


@dataclass
class QuestionGroup:
    number: Optional[int]                          # detected Q number (may be None)
    regions: List[Tuple[LayoutBlock, List[Block]]] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────

def segment(
    regions: List[Tuple[LayoutBlock, List[Block]]],
) -> Tuple[List[Block], List[QuestionGroup]]:
    """
    Group (LayoutBlock, List[Block]) pairs into per-question clusters.

    Parameters
    ----------
    regions : ordered list of (layout_block, block_list) from the OCR pipeline

    Returns
    -------
    ungrouped : blocks before the first question number was found
    groups    : one QuestionGroup per detected question
    """
    ungrouped: List[Block]         = []
    groups:    List[QuestionGroup] = []
    current:   Optional[QuestionGroup] = None

    prev_bbox_y2: Optional[int] = None

    for lb, blocks in regions:
        if not blocks:
            prev_bbox_y2 = lb.bbox[3]
            continue

        qnum = _detect_question_number(blocks)

        # Also check for a large y-gap from the previous block
        # (only applies once we are already inside a question group)
        gap_break = False
        if (
            current is not None
            and prev_bbox_y2 is not None
            and lb.bbox[1] - prev_bbox_y2 > _Y_GAP_THRESHOLD
            and qnum is None
        ):
            gap_break = True

        if qnum is not None or gap_break:
            if current is not None:
                groups.append(current)
            current = QuestionGroup(number=qnum, regions=[(lb, blocks)])
        elif current is None:
            # Before the first question number was encountered
            ungrouped.extend(blocks)
        else:
            current.regions.append((lb, blocks))

        prev_bbox_y2 = lb.bbox[3]

    if current is not None:
        groups.append(current)

    return ungrouped, groups


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_question_number(blocks: List[Block]) -> Optional[int]:
    """
    Scan the first few blocks of a region.
    Return the detected integer question number, or None.
    """
    checked = 0
    for block in blocks:
        if isinstance(block, TextBlock):
            m = _Q_START.match(block.value)
            if m:
                return int(m.group(1))
            checked += 1
            if checked >= _LOOKAHEAD:
                break
    return None
