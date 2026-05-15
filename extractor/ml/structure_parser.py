"""
Structure Parser
================
Converts a flat list of ocr_blocks (as stored in DocumentPage.ocr_blocks)
into a structured_content JSON document suitable for the editor panel.

Node types produced:
  section   — from title blocks
  paragraph — from plain_text blocks (may merge consecutive ones)
  table     — from table blocks
  image     — from figure blocks

Each node carries source_block_ids: the list of block_id values from the
originating ocr_blocks, enabling two-way sync between canvas and editor.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


def _extract_text(blocks: List[Dict]) -> str:
    """Return the leading text value from a blocks list, for pattern matching."""
    for b in blocks:
        if b.get("type") == "text":
            return b["value"]
    return ""


def _inline_content(blocks: List[Dict]) -> List[Dict]:
    """
    Convert a list of raw Block dicts (text/latex/image/table/chemical)
    into the inline_content format used by the structured editor.
    Only text and latex blocks are meaningful as inline; others are dropped
    (they will appear as their own top-level nodes).

    Consecutive text blocks are merged with a space so the editor does not
    show disconnected word-chunks (OCR often emits one TextBlock per detected
    text element on the same line).
    """
    result = []
    for b in blocks:
        t = b.get("type")
        if t == "text":
            text = b["value"]
            if result and result[-1]["type"] == "text":
                # Merge into the previous text block instead of creating a new one
                result[-1]["value"] = result[-1]["value"] + " " + text
            else:
                result.append({"type": "text", "value": text})
        elif t == "latex":
            result.append({"type": "latex", "value": b["value"], "display": b.get("display", False)})
    return result


def _make_id() -> str:
    return str(uuid.uuid4())


def parse(ocr_blocks: List[Dict]) -> Dict:
    """
    Parse a flat list of ocr_block dicts into a structured_content document.

    Parameters
    ----------
    ocr_blocks : list of dicts, as stored in DocumentPage.ocr_blocks
                 Each dict has: block_id, label, bbox, reading_order,
                 column_idx, blocks (list of Block dicts), duration_ms

    Returns
    -------
    dict  — structured_content JSON matching the editor schema
    """
    sorted_blocks = sorted(ocr_blocks, key=lambda b: b.get("reading_order", 0))
    top_nodes: List[Dict] = []
    current_section: Optional[Dict] = None

    def emit(node: Dict) -> None:
        if current_section is not None:
            current_section["children"].append(node)
        else:
            top_nodes.append(node)

    for ocr_block in sorted_blocks:
        label = ocr_block.get("label", "")
        block_id = ocr_block.get("block_id", "")
        inner_blocks = ocr_block.get("blocks", [])

        # OCR failed for this block — emit a visible error node instead of dropping it
        if ocr_block.get("error") and not inner_blocks:
            emit({
                "id": _make_id(),
                "type": "error",
                "source_block_ids": [block_id],
                "label": label,
                "message": str(ocr_block["error"]),
            })
            continue

        # ── figure ────────────────────────────────────────────────────────────
        if label == "figure":
            img = next((b for b in inner_blocks if b.get("type") == "image"), None)
            emit({
                "id": _make_id(),
                "type": "image",
                "source_block_ids": [block_id],
                "url": img["url"] if img else "",
                "alt": img.get("alt", "") if img else "",
                "width": img.get("width", 0) if img else 0,
                "height": img.get("height", 0) if img else 0,
            })
            continue

        # ── table ─────────────────────────────────────────────────────────────
        if label == "table":
            tbl = next((b for b in inner_blocks if b.get("type") == "table"), None)
            emit({
                "id": _make_id(),
                "type": "table",
                "source_block_ids": [block_id],
                "rows": tbl["rows"] if tbl else 0,
                "cols": tbl["cols"] if tbl else 0,
                "cells": tbl["cells"] if tbl else [],
            })
            continue

        # ── isolate_formula ────────────────────────────────────────────────────
        if label == "isolate_formula":
            latex_b = next((b for b in inner_blocks if b.get("type") == "latex"), None)
            emit({
                "id": _make_id(),
                "type": "paragraph",
                "source_block_ids": [block_id],
                "content": [{"type": "latex", "value": latex_b["value"] if latex_b else "", "display": True}],
            })
            continue

        # ── title → section ───────────────────────────────────────────────────
        if label == "title":
            new_section: Dict = {
                "id": _make_id(),
                "type": "section",
                "source_block_ids": [block_id],
                "heading": _inline_content(inner_blocks),
                "children": [],
            }
            top_nodes.append(new_section)
            current_section = new_section
            continue

        # ── plain_text → paragraph ─────────────────────────────────────────────
        emit({
            "id": _make_id(),
            "type": "paragraph",
            "source_block_ids": [block_id],
            "content": _inline_content(inner_blocks),
        })

    return {"version": 1, "nodes": top_nodes}
