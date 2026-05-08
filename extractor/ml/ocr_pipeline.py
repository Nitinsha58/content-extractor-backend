"""
OCR Orchestrator
================
Routes each LayoutBlock to the correct pix2text OCR component and
converts the output into standardised Block objects (from schema.py).

Routing table
─────────────
  plain_text / title  →  TextFormulaOCR  →  TextBlock + LatexBlock (inline/display)
  isolate_formula     →  LatexOCR        →  LatexBlock (display=True)
  table               →  TableOCR        →  TableBlock  (falls back to text on error)
  figure              →  ImageHandler    →  ImageBlock
  anything else       →  TextFormulaOCR  →  text fallback

Formula config mirrors FinalTest/main.py exactly (ONNX backend, perfected MFR).
LaTeX strings are cleaned through FinalTest/cleaner.py (single source of truth).

Call init_models() once at app startup to pre-load all weights.
"""

from __future__ import annotations

import os
import re
import sys
from typing import List, Optional

# ── Fix: rapidocr TextRecognizer accesses cfg.font_path but cnocr never sets it
# Patch DEFAULT_CFG before any CnOcr instance is created (happens in init_models)
try:
    from cnocr.ppocr.rapid_recognizer import Config as _RapidConfig
    if "font_path" not in _RapidConfig.DEFAULT_CFG:
        _RapidConfig.DEFAULT_CFG["font_path"] = None
except Exception:
    pass

from PIL import Image

# ── Import clean_latex from cleaner module — copied from FinalTest ────────────────
from cleaner import clean_latex  # noqa: E402

from image_handler import save_crop
from layout import LayoutBlock
from schema import Block, LatexBlock, TableBlock, TextBlock

# ── Model singletons (lazy-loaded) ───────────────────────────────────────────
_text_formula_ocr = None   # TextFormulaOCR — plain_text (MFD + text OCR)
_text_only_ocr    = None   # TextFormulaOCR with enable_formula=False — titles
_p2t_model:        Optional[object] = None   # Pix2Text — for isolate_formula
_table_ocr:        Optional[object] = None   # False = failed to load


# ── Formula config matching FinalTest ────────────────────────────────────────
_FORMULA_CFG = {
    "model_backend": "onnx",
    "model_kwargs": {"decoder_file_name": "decoder_model.onnx"},
}


def init_models() -> None:
    """Pre-load all OCR models. Call once at server startup."""
    _get_text_formula_ocr()
    _get_text_only_ocr()
    _get_p2t_model()
    # TableOCR is lazy — don't block startup on it


# ── Getters ───────────────────────────────────────────────────────────────────

def _get_text_formula_ocr():
    global _text_formula_ocr
    if _text_formula_ocr is None:
        from pix2text import TextFormulaOCR
        _text_formula_ocr = TextFormulaOCR.from_config(
            total_configs={
                "languages": ("en",),
                "mfd":     {"model_name": "mfd-1.5", "model_backend": "onnx"},
                "text":    {},
                "formula": _FORMULA_CFG,
            },
            enable_formula=True,
            enable_spell_checker=False,
        )
    return _text_formula_ocr


def _get_text_only_ocr():
    """TextFormulaOCR with formula detection OFF — for title blocks.
    Shared CnOCR text engine, no MFD overhead (~3× faster than full pipeline).
    """
    global _text_only_ocr
    if _text_only_ocr is None:
        from pix2text import TextFormulaOCR
        _text_only_ocr = TextFormulaOCR.from_config(
            total_configs={
                "languages": ("en",),
                "text":    {},
            },
            enable_formula=False,
            enable_spell_checker=False,
        )
    return _text_only_ocr


def _get_p2t_model():
    """Pix2Text singleton — same config as FinalTest/main.py."""
    global _p2t_model
    if _p2t_model is None:
        from pix2text import Pix2Text
        _p2t_model = Pix2Text.from_config(formula_config=_FORMULA_CFG)
    return _p2t_model


def _get_table_ocr():
    """
    Returns the Pix2Text table_ocr instance with tuned structure detection thresholds.
    Uses the main Pix2Text initialization which handles table extraction.

    Threshold tuning:
    - row/column detection: lowered to 0.3 (from 0.5) to detect more rows/columns
    - This improves recall for tables with uncertain boundaries
    """
    global _table_ocr
    if _table_ocr is None:
        try:
            from pix2text import Pix2Text, TableOCR
            p2t = Pix2Text(enable_table=True)
            _table_ocr = p2t.table_ocr

            # If table_ocr is available, re-create with tuned thresholds
            if _table_ocr is not None and hasattr(_table_ocr, 'text_ocr'):
                try:
                    structure_thresholds = {
                        'table': 0.5,
                        'table column': 0.3,            # lower: detect more columns
                        'table row': 0.3,               # lower: detect more rows
                        'table column header': 0.4,
                        'table projected row header': 0.3,
                        'table spanning cell': 0.5,
                        'no object': 10,
                    }
                    _table_ocr = TableOCR(
                        text_ocr=_table_ocr.text_ocr,
                        spellchecker=getattr(_table_ocr, 'spellchecker', None),
                        structure_thresholds=structure_thresholds,
                        threshold_percentage=0.08,      # slightly lower for better row/col detection
                    )
                except Exception:
                    # If re-creation fails, use the default table_ocr
                    pass

            # If table_ocr is still None, fall back to using Pix2Text directly
            if _table_ocr is None:
                _table_ocr = p2t
        except Exception:
            _table_ocr = False   # permanent failure sentinel

    return _table_ocr if _table_ocr else None


# ── Main entry point ──────────────────────────────────────────────────────────

def process_region(lb: LayoutBlock) -> List[Block]:
    """
    Convert one LayoutBlock into a list of Block objects.
    Never raises — returns [] on unrecoverable failure.
    """
    label = lb.label
    crop  = lb.crop

    if label == "plain_text":
        return _process_text(crop)
    elif label == "title":
        return _process_text(crop, formula=False)
    elif label == "isolate_formula":
        return _process_formula(crop)
    elif label == "table":
        return _process_table(crop)
    elif label == "figure":
        return [save_crop(crop, alt="figure")]
    else:
        # Unknown label — best-effort text extraction
        return _process_text(crop)


# ── Per-modality processors ───────────────────────────────────────────────────

def _process_text(crop: Image.Image, formula: bool = True) -> List[Block]:
    """
    TextFormulaOCR with return_text=False → typed element list.
    formula=False uses the no-MFD model (faster, for title blocks).
    Converts each element to a TextBlock (text) or LatexBlock (formula).

    When formula=True and MFD misidentifies the entire region as a formula
    (all output blocks are LaTeX, no text), falls back to text-only OCR so
    that plain text crops relabeled from formula regions are handled correctly.
    """
    ocr = _get_text_formula_ocr() if formula else _get_text_only_ocr()
    try:
        result = ocr.recognize(crop, return_text=False)
    except Exception:
        return []

    # Fallback: model might return a plain string even with return_text=False
    if isinstance(result, str):
        return [TextBlock(value=result.strip())] if result.strip() else []

    if not result:
        return []

    # Sort by (line_number, x-position) for correct inline reading order
    def _sort_key(e):
        line = e.get("line_number") or 0
        pos  = e.get("position")
        try:
            import numpy as _np
            arr = _np.array(pos)
            x0 = float(arr.flat[0]) if arr.size > 0 else 0.0
        except Exception:
            x0 = 0.0
        return (line, x0)

    elements = sorted(result, key=_sort_key)

    blocks: List[Block] = []
    for elem in elements:
        elem_type = elem.get("type", "text")
        text      = (elem.get("text") or "").strip()
        if not text:
            continue

        if elem_type == "text":
            blocks.append(TextBlock(value=text))
        elif elem_type in ("embedding", "isolated"):
            blocks.append(
                LatexBlock(
                    value=clean_latex(text),
                    display=(elem_type == "isolated"),
                )
            )

    # If formula mode returned only LaTeX blocks (no text), MFD likely
    # misidentified the entire region as a formula. Re-run with text-only OCR.
    if formula and blocks and all(isinstance(b, LatexBlock) for b in blocks):
        return _process_text(crop, formula=False)

    return blocks


def _process_formula(crop: Image.Image) -> List[Block]:
    """
    Pix2Text.recognize_formula on an isolated display formula region.
    Returns a single LatexBlock(display=True).  Raises on failure so the
    caller (OcrView) can record a structured error block.
    """
    ocr = _get_p2t_model()
    result = ocr.recognize_formula(crop)   # let exceptions propagate
    text = clean_latex(str(result).strip())
    if not text:
        raise RuntimeError("Formula recognition returned empty result")
    return [LatexBlock(value=text, display=True)]


def _reconstruct_table_from_blocks(blocks: List[Block], img_size: tuple) -> Optional[TableBlock]:
    """
    Reconstruct table structure from text blocks using date pattern detection.
    Groups blocks into rows based on detecting actual date patterns (MM/DD/YYYY).
    """
    if not blocks or len(blocks) < 2:
        return None

    # Extract text blocks only
    text_blocks = [b for b in blocks if isinstance(b, TextBlock)]
    if len(text_blocks) < 2:
        return None

    # Helper: detect if text is a date in MM/DD/YYYY format
    def is_date_pattern(text: str) -> bool:
        text = text.strip()
        # Match MM/DD/YYYY, MM/DD/YY, M/D/YYYY, etc
        date_patterns = [
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',  # MM/DD/YYYY
            r'^\d{1,2}-\d{1,2}-\d{2,4}$',  # MM-DD-YYYY
        ]
        for pattern in date_patterns:
            if re.match(pattern, text):
                return True
        return False

    # Group blocks into rows by detecting date patterns
    rows: List[List[TextBlock]] = []
    current_row: List[TextBlock] = []

    for block in text_blocks:
        text = block.value.strip()
        if is_date_pattern(text):
            # Start a new row with this date
            if current_row:
                rows.append(current_row)
            current_row = [block]
        else:
            # Add to current row
            current_row.append(block)

    # Don't forget the last row
    if current_row:
        rows.append(current_row)

    if len(rows) < 2:
        # Not enough rows to be a table
        return None

    # Determine the expected number of columns (most common row length)
    row_lengths = [len(row) for row in rows]
    from collections import Counter
    col_count = Counter(row_lengths).most_common(1)[0][0] if row_lengths else 0
    
    if col_count < 2:
        # Not enough columns
        return None

    # Normalize all rows to have the same column count
    grid: List[List[List[Block]]] = []
    for row in rows:
        grid_row: List[List[Block]] = []
        for block in row:
            grid_row.append([block])
        
        # Pad with empty cells to match expected column count
        while len(grid_row) < col_count:
            grid_row.append([])
        
        # Truncate if too many columns
        grid_row = grid_row[:col_count]
        grid.append(grid_row)

    return TableBlock(rows=len(grid), cols=col_count, cells=grid)


def _process_table(crop: Image.Image) -> List[Block]:
    """
    Attempt to extract table structure using multiple strategies.
    1. Try TableOCR with markdown output (fast path)
    2. Try TableOCR with cell parsing (slower, more detailed)
    Raises RuntimeError on all failure paths — caller records a structured error block.
    """
    table_ocr = _get_table_ocr()

    if table_ocr is None or not hasattr(table_ocr, 'recognize'):
        raise RuntimeError("Table OCR model unavailable")

    try:
        raw = table_ocr.recognize(crop, out_cells=True, out_markdown=True)
    except Exception as e:
        raise RuntimeError(f"Table OCR failed: {e}") from e

    # Fast path: pre-rendered markdown
    if raw.get('markdown') and len(raw['markdown']) > 0:
        markdown_str = raw['markdown'][0].strip()
        if markdown_str:
            result = _markdown_to_table_block(markdown_str)
            if result.rows > 0 and result.cols > 0:
                return [result]

    # Cell-parsing path
    result = _build_table_block(raw)
    if result.rows > 0 and result.cols > 0:
        return [result]

    raise RuntimeError("Table OCR returned empty structure")


def _markdown_to_table_block(markdown_str: str) -> TableBlock:
    """
    Parse a markdown table string into a TableBlock.
    Format:
      | Header1 | Header2 |
      |---------|---------|
      | Data1   | Data2   |
    """
    lines = [line.strip() for line in markdown_str.split('\n') if line.strip()]
    if len(lines) < 3:
        return TableBlock(rows=0, cols=0, cells=[])

    def parse_row(line: str) -> List[str]:
        # Remove leading/trailing pipes and split by pipe
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        cells = [cell.strip() for cell in line.split('|')]
        return [c for c in cells if c]  # remove empty

    try:
        # Parse header row
        header_row = parse_row(lines[0])
        if not header_row:
            return TableBlock(rows=0, cols=0, cells=[])

        num_cols = len(header_row)

        # Skip separator row (line[1])
        # Parse data rows
        grid: List[List[List[Block]]] = []
        for line in lines[2:]:
            if '---' in line:  # skip extra separator rows
                continue
            row = parse_row(line)
            if not row:
                continue
            # Pad row to match column count
            while len(row) < num_cols:
                row.append('')
            grid_row = [[TextBlock(value=cell)] if cell else [] for cell in row[:num_cols]]
            grid.append(grid_row)

        if grid:
            # Prepend header row
            header_blocks = [[TextBlock(value=cell)] for cell in header_row]
            grid.insert(0, header_blocks)

        return TableBlock(rows=len(grid), cols=num_cols, cells=grid)
    except Exception:
        return TableBlock(rows=0, cols=0, cells=[])


def _build_table_block(result) -> TableBlock:
    """
    Convert TableOCR output into a TableBlock.

    Supported formats:
    ─────────────────
    1. Pix2Text format (actual):  {"cells": [[{row_nums, column_nums, cell text}, …], ...]}
       outer list = tables, inner list = cells for that table (flat, not 2-D grid)
    2. Legacy format 1:           {"cells": [{row_idx, col_idx, text}, …]}
    3. Legacy format 2:           [[cell_str, …], …] (2-D list)
    """
    if isinstance(result, dict):
        cells_raw = result.get("cells", result.get("rows", []))
    elif isinstance(result, list):
        cells_raw = result
    else:
        return TableBlock(rows=0, cols=0, cells=[])

    if not cells_raw:
        return TableBlock(rows=0, cols=0, cells=[])

    # ── Format 3: Pix2Text format (List[List[dict]] with row_nums/column_nums) ──
    if isinstance(cells_raw[0], (list, tuple)) and len(cells_raw[0]) > 0:
        if isinstance(cells_raw[0][0], dict) and "row_nums" in cells_raw[0][0]:
            # cells_raw[0] = flat list of cell dicts for the first (main) table
            table_cells = cells_raw[0]
            if not table_cells:
                return TableBlock(rows=0, cols=0, cells=[])

            # Find grid dimensions
            try:
                max_row = max(c.get("row_nums", [0])[0] for c in table_cells) + 1
                max_col = max(c.get("column_nums", [0])[0] for c in table_cells) + 1
            except (ValueError, IndexError, TypeError):
                return TableBlock(rows=0, cols=0, cells=[])

            # Create empty grid
            grid: List[List[List]] = [[[] for _ in range(max_col)] for _ in range(max_row)]

            # Populate grid with cell text
            for cell in table_cells:
                try:
                    row_indices = cell.get("row_nums", [])
                    col_indices = cell.get("column_nums", [])
                    if not row_indices or not col_indices:
                        continue

                    r = row_indices[0]
                    c = col_indices[0]
                    text = (cell.get("cell text") or "").strip()

                    if 0 <= r < max_row and 0 <= c < max_col:
                        if text:
                            grid[r][c] = [TextBlock(value=text)]
                        # Note: empty cells remain as []
                except (TypeError, KeyError, IndexError):
                    continue

            return TableBlock(rows=max_row, cols=max_col, cells=grid)

    # ── Format 1: key-value with row_idx / col_idx ────────────────────────────
    if isinstance(cells_raw[0], dict) and "row_idx" in cells_raw[0]:
        max_row = max(c.get("row_idx", 0) for c in cells_raw) + 1
        max_col = max(c.get("col_idx", 0) for c in cells_raw) + 1
        grid: List[List[List]] = [[[] for _ in range(max_col)] for _ in range(max_row)]
        for cell in cells_raw:
            r    = cell.get("row_idx", 0)
            c    = cell.get("col_idx", 0)
            text = (cell.get("text") or cell.get("content") or "").strip()
            if r < max_row and c < max_col and text:
                grid[r][c] = [TextBlock(value=text)]
        return TableBlock(rows=max_row, cols=max_col, cells=grid)

    # ── Format 2: 2-D list ────────────────────────────────────────────────────
    if isinstance(cells_raw[0], (list, tuple)):
        grid = []
        max_col = 0
        for row in cells_raw:
            grid_row = []
            for cell in row:
                if isinstance(cell, dict):
                    text = (cell.get("text") or "").strip()
                else:
                    text = str(cell).strip()
                grid_row.append([TextBlock(value=text)] if text else [])
            grid.append(grid_row)
            max_col = max(max_col, len(grid_row))
        return TableBlock(rows=len(grid), cols=max_col, cells=grid)

    return TableBlock(rows=0, cols=0, cells=[])
