"""
OCR Orchestrator
================
Routes each LayoutBlock to the correct pix2text OCR component and
converts the output into standardised Block objects (from schema.py).

Routing table
─────────────
  plain_text / title  →  TextFormulaOCR  →  TextBlock + LatexBlock (inline/display)
  isolate_formula     →  LatexOCR        →  LatexBlock (display=True)
  table               →  TextFormulaOCR + table_structure  →  TableBlock
                         (error node if table_structure is absent — see ADR-0002)
  figure              →  ImageHandler    →  ImageBlock
  anything else       →  TextFormulaOCR  →  text fallback

Formula config mirrors FinalTest/main.py exactly (ONNX backend, perfected MFR).
LaTeX strings are cleaned through FinalTest/cleaner.py (single source of truth).

Call init_models() once at app startup to pre-load all weights.
"""

from __future__ import annotations

import os
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

import re

from PIL import Image

# ── Import clean_latex from cleaner module — copied from FinalTest ────────────────
from cleaner import clean_latex  # noqa: E402

from image_handler import save_crop
from layout import LayoutBlock
from schema import Block, LatexBlock, TableBlock, TextBlock

# ── False-positive formula detector ──────────────────────────────────────────
# pix2text MFD sometimes tags italic/roman word characters as inline formulas,
# producing \mathrm{o f}\mathrm{f}... instead of plain text.  A LaTeX value
# that consists *entirely* of \mathrm{letters/spaces} groups is always plain
# text — real math expressions contain digits, operators, or other commands.
_PURE_MATHRM_RE = re.compile(r'^(\s*\\mathrm\{[A-Za-z][A-Za-z\s]*\}\s*)+$')
_MATHRM_CONTENT_RE = re.compile(r'\\mathrm\{([^}]+)\}')


def _mathrm_to_plain_text(latex: str) -> Optional[str]:
    """Return extracted plain text if latex is a pure-\\mathrm false-positive, else None."""
    if _PURE_MATHRM_RE.match(latex):
        parts = [m.group(1).strip() for m in _MATHRM_CONTENT_RE.finditer(latex)]
        joined = ' '.join(p for p in parts if p)
        return joined or None
    return None

# ── Model singletons (lazy-loaded) ───────────────────────────────────────────
_text_formula_ocr = None   # TextFormulaOCR — plain_text (MFD + text OCR)
_text_only_ocr    = None   # TextFormulaOCR with enable_formula=False — titles
_p2t_model:        Optional[object] = None   # Pix2Text — for isolate_formula


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
        # MFD default conf=0.25 causes false-positive formula detections on
        # parenthetical notation like "(T)" in plain text, which masks that
        # region before CnOCR runs and garbles the surrounding text.
        # Raise to 0.45 to reduce false positives while still catching real
        # inline/display formulas (which typically score >> 0.5).
        if hasattr(_text_formula_ocr, 'mfd') and _text_formula_ocr.mfd is not None:
            _orig_detect = _text_formula_ocr.mfd.detect
            def _detect_high_conf(img_list, resized_shape=768, box_margin=0, conf=0.45, **kw):
                return _orig_detect(img_list, resized_shape=resized_shape, box_margin=box_margin, conf=conf, **kw)
            _text_formula_ocr.mfd.detect = _detect_high_conf
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
        _p2t_model = Pix2Text.from_config(
            total_configs={"languages": ("en",)},
            formula_config=_FORMULA_CFG,
        )
    return _p2t_model


# ── Main entry point ──────────────────────────────────────────────────────────

def process_region(lb: LayoutBlock) -> List[Block]:
    """
    Convert one LayoutBlock into a list of Block objects.
    Never raises — returns [] on unrecoverable failure.
    """
    label = lb.label
    crop  = lb.crop

    if label == "table":
        # Do NOT pad the outer table crop — row_dividers / col_dividers are
        # fractions of the original crop dimensions; padding would shift all
        # cell boundaries. Padding is applied per-cell in _ocr_cell_typed.
        return _process_table(crop, lb.table_structure)

    crop = _pad_crop(crop, label)

    if label == "plain_text":
        return _process_text(crop)
    elif label == "title":
        return _process_text(crop, formula=False)
    elif label == "isolate_formula":
        return _process_formula(crop)
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
            cleaned = clean_latex(text)
            plain = _mathrm_to_plain_text(cleaned) if elem_type == "embedding" else None
            if plain is not None:
                blocks.append(TextBlock(value=plain))
            else:
                blocks.append(LatexBlock(value=cleaned, display=(elem_type == "isolated")))

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


_CELL_PADDING = 4    # pixels of padding added around each cell crop
_CELL_MIN_DIM = 32   # cells smaller than this get upscaled before OCR


def _pad_crop(crop: Image.Image, label: str) -> Image.Image:
    """
    Add white padding around a crop and optionally upscale it before OCR.

    Padding rules (dynamic, based on the smaller crop dimension):
      isolate_formula         → max(16, min_dim × 0.15)
      plain_text / title      → max(12, min_dim × 0.08)
      figure                  → max(1,  min_dim × 0.05)
      everything else         → max(15, min_dim × 0.10)

    If the padded height is still < 80 px the crop is upscaled 2×–3× so
    OCR models have enough resolution to resolve superscripts, subscripts,
    and small glyphs reliably.
    """
    import cv2
    import numpy as np

    w, h = crop.size
    min_dim = min(w, h)

    if label == "isolate_formula":
        pad = max(16, int(min_dim * 0.15))
    elif label in ("plain_text", "title"):
        pad = max(12, int(min_dim * 0.08))
    elif label == "figure":
        pad = max(1, int(min_dim * 0.05))
    else:
        pad = max(15, int(min_dim * 0.10))

    arr = np.array(crop.convert("RGB"), dtype=np.uint8)
    arr = cv2.copyMakeBorder(
        arr, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    padded_h = arr.shape[0]
    if padded_h < 80:
        # Scale to reach ~80 px, but keep factor in the 2×–3× range.
        scale = min(3.0, max(2.0, 80.0 / padded_h))
        new_w = max(1, round(arr.shape[1] * scale))
        new_h = max(1, round(padded_h * scale))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    return Image.fromarray(arr)


def _process_table(
    crop: Image.Image,
    table_structure: Optional[dict],
) -> List[Block]:
    """
    Stage 3 of the three-stage table pipeline (ADR-0002, ADR-0003).

    Requires table_structure with finalized=True from a prior Finalize action
    (POST /api/debug/table-cell-types/).  Each cell is routed to the right
    pix2text handler based on its Cell Type (plain_text / title /
    isolate_formula / figure).

    Raises RuntimeError if table_structure is absent or not finalized.
    """
    if not table_structure:
        raise RuntimeError(
            "table_structure is required for table OCR. "
            "Run Table Structure Analysis, then click Finalize."
        )

    if not table_structure.get("finalized", False):
        raise RuntimeError(
            "Table structure must be finalized before OCR can run. "
            "Click 'Finalize' on the table block to lock the grid and detect cell types."
        )

    # Apply the same skew correction that was used during TATR analysis.
    deskew_angle = table_structure.get("deskew_angle", 0.0)
    if deskew_angle:
        from .tatr import deskew_crop as _deskew_crop
        crop, _ = _deskew_crop(crop, angle=deskew_angle)

    crop_w, crop_h = crop.size
    row_dividers = table_structure.get("row_dividers", [])
    col_dividers = table_structure.get("col_dividers", [])

    row_boundaries = [0] + [round(d * crop_h) for d in row_dividers] + [crop_h]
    col_boundaries = [0] + [round(d * crop_w) for d in col_dividers] + [crop_w]

    n_rows = len(row_boundaries) - 1
    n_cols = len(col_boundaries) - 1

    if n_rows == 0 or n_cols == 0:
        raise RuntimeError("table_structure contains no rows or columns")

    cell_types = table_structure.get("cell_types") or []
    grid: List[List[List[Block]]] = [[[] for _ in range(n_cols)] for _ in range(n_rows)]

    for r in range(n_rows):
        for c in range(n_cols):
            cell_crop = _crop_cell(crop, row_boundaries, col_boundaries, r, c)
            if cell_crop is None:
                continue
            cell_type = (
                cell_types[r][c]
                if r < len(cell_types) and c < len(cell_types[r])
                else "plain_text"
            )
            grid[r][c] = _ocr_cell_typed(cell_crop, cell_type)

    return [TableBlock(rows=n_rows, cols=n_cols, cells=grid)]


def _ocr_cell_typed(cell_crop: Image.Image, cell_type: str) -> List[Block]:
    """Route a single cell crop to the right OCR handler by its Cell Type."""
    cell_crop = _pad_crop(cell_crop, cell_type)
    if cell_type == "title":
        return _ocr_cell(_get_text_only_ocr(), cell_crop)
    elif cell_type == "isolate_formula":
        try:
            return _process_formula(cell_crop)
        except Exception:
            return _ocr_cell(_get_text_formula_ocr(), cell_crop)
    elif cell_type == "figure":
        return [save_crop(cell_crop, alt="figure")]
    else:  # plain_text or unrecognised default
        return _ocr_cell(_get_text_formula_ocr(), cell_crop)


def _crop_cell(
    table_crop: Image.Image,
    row_boundaries: list,
    col_boundaries: list,
    r: int,
    c: int,
) -> Optional[Image.Image]:
    """Crop one cell with padding, clamped to the table image bounds."""
    tw, th = table_crop.size
    x1 = max(0, col_boundaries[c]     - _CELL_PADDING)
    y1 = max(0, row_boundaries[r]     - _CELL_PADDING)
    x2 = min(tw, col_boundaries[c + 1] + _CELL_PADDING)
    y2 = min(th, row_boundaries[r + 1] + _CELL_PADDING)
    if x2 <= x1 or y2 <= y1:
        return None
    cell = table_crop.crop((x1, y1, x2, y2)).convert("RGB")
    # Upscale very small cells so OCR has enough resolution to work with
    cw, ch = cell.size
    if cw < _CELL_MIN_DIM or ch < _CELL_MIN_DIM:
        scale = max(_CELL_MIN_DIM / max(cw, 1), _CELL_MIN_DIM / max(ch, 1), 1.0)
        cell = cell.resize((max(1, round(cw * scale)), max(1, round(ch * scale))), Image.LANCZOS)
    return cell


def _ocr_cell(ocr, cell_crop: Image.Image) -> List:
    """Run TextFormulaOCR on a single cell crop and return a list of Blocks."""
    try:
        result = ocr.recognize(cell_crop, return_text=False)
    except Exception:
        return []

    if isinstance(result, str):
        text = result.strip()
        return [TextBlock(value=text)] if text else []

    blocks: List = []
    # Sort elements top→bottom, left→right for natural reading order within the cell
    elems = []
    cw, ch = cell_crop.size
    for elem in (result or []):
        text = (elem.get("text") or "").strip()
        if not text:
            continue
        top_y, left_x = _elem_top_left(elem.get("position"), cw, ch)
        elems.append((top_y, left_x, elem))
    elems.sort(key=lambda t: (t[0], t[1]))

    for _, _, elem in elems:
        elem_type = elem.get("type", "text")
        text = (elem.get("text") or "").strip()
        if not text:
            continue
        if elem_type == "text":
            if blocks and blocks[-1].type == "text":
                blocks[-1] = TextBlock(value=blocks[-1].value + " " + text)
            else:
                blocks.append(TextBlock(value=text))
        elif elem_type in ("embedding", "isolated"):
            blocks.append(LatexBlock(value=clean_latex(text), display=(elem_type == "isolated")))

    return blocks


# ── Table geometry helpers ────────────────────────────────────────────────────

def _elem_top_left(position, default_w: int, default_h: int) -> tuple:
    """Return (top_y, left_x) from a pix2text position polygon."""
    try:
        import numpy as _np
        arr = _np.array(position).reshape(-1, 2)
        return float(arr[:, 1].min()), float(arr[:, 0].min())
    except Exception:
        return default_h / 2.0, default_w / 2.0


