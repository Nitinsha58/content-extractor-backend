"""
ExamLayoutParser
================
Uses DocLayout-YOLO (doclayout_yolo_docstructbench_imgsz1024.pt) for layout
detection — purpose-trained on document structure, much better formula/figure
separation than the generic cnstd yolov7_tiny.

Adds on top:
  - Two-column detection (histogram of x-midpoints → gap in 35–65% of page width)
  - Canonical reading order: full left column top-to-bottom, then right column
  - Returns LayoutBlock dataclasses with the image crop already extracted

Run on a single PIL Image:
    parser = ExamLayoutParser()
    blocks = parser.parse(img)   # → List[LayoutBlock]
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

# ── Weights path ──────────────────────────────────────────────────────────────
# Prefer pix2text's cached copy (downloaded automatically by download_models.py).
# Fall back to the local LayoutAnalyser/ copy for machines that have it already.
_HERE         = Path(os.path.dirname(os.path.abspath(__file__)))
_P2T_WEIGHTS  = Path.home() / ".pix2text" / "1.1" / "layout-docyolo" / "doclayout_yolo_docstructbench_imgsz1024.pt"
_LOCAL_WEIGHTS = _HERE / ".." / "LayoutAnalyser" / "weights.pt"
_WEIGHTS_PT   = _P2T_WEIGHTS if _P2T_WEIGHTS.exists() else _LOCAL_WEIGHTS

# ── DocLayout-YOLO class id → our label string ────────────────────────────────
# Classes from DocLayout-YOLO-DocStructBench (10 classes):
_DOCLAYOUT_LABEL: dict[int, str] = {
    0: "title",
    1: "plain_text",
    2: "abandon",           # headers, footers, page numbers  → skip
    3: "figure",
    4: "plain_text",        # figure_caption → treat as text
    5: "table",
    6: "title",             # table_caption  → treat as heading
    7: "plain_text",        # table_footnote → treat as text
    8: "isolate_formula",
    9: "plain_text",        # formula_caption → treat as text
}

# Labels that contribute no useful content → skip entirely
_SKIP_LABELS = {"abandon"}


@dataclass
class LayoutBlock:
    label: str                        # "title" | "plain_text" | "isolate_formula" | "table" | "figure"
    bbox: List[int]                   # [x1, y1, x2, y2] in pixels (absolute)
    crop: Image.Image                 # pre-extracted region
    column_idx: int                   # 0 = left / single column,  1 = right column
    reading_order: int                # final sort index (0-based)
    confidence: float
    table_structure: Optional[dict] = None  # populated by TATR before OCR runs


class ExamLayoutParser:
    """
    Layout detection with two-column awareness for exam paper images.
    Uses DocLayout-YOLO (imgsz=1024) for high-accuracy document structure detection.

    Usage::
        parser = ExamLayoutParser()
        blocks = parser.parse(img)
    """

    def __init__(self) -> None:
        self._model = None   # doclayout_yolo.YOLOv10, lazy-loaded

    # ── Public API ─────────────────────────────────────────────────────────────

    def parse(self, img: Image.Image) -> List[LayoutBlock]:
        """
        Run layout detection on *img*.
        Returns blocks in reading order (left column → right column, top → bottom).
        """
        self._load_model()
        img_rgb = img.convert("RGB")
        img_w, img_h = img_rgb.size

        results = self._model.predict(
            img_rgb,
            imgsz=1024,
            conf=0.35,
            iou=0.40,
            verbose=False,
        )

        raw: list[dict] = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label  = _DOCLAYOUT_LABEL.get(cls_id, "plain_text")
                if label in _SKIP_LABELS:
                    continue

                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                # Clamp to image bounds and drop degenerate boxes
                x1, x2 = max(0.0, x1), min(float(img_w), x2)
                y1, y2 = max(0.0, y1), min(float(img_h), y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                bbox = [int(x1), int(y1), int(x2), int(y2)]
                crop = img_rgb.crop((int(x1), int(y1), int(x2), int(y2)))

                raw.append({
                    "label":  label,
                    "bbox":   bbox,
                    "crop":   crop,
                    "conf":   conf,
                    "x_mid":  (x1 + x2) / 2.0,
                    "y_mid":  (y1 + y2) / 2.0,
                })

        if not raw:
            return []

        # ── Two-column split ──────────────────────────────────────────────────
        col_boundary = self._detect_column_boundary(raw, img_w)
        for b in raw:
            b["column_idx"] = (
                0 if col_boundary is None or b["x_mid"] < col_boundary else 1
            )

        # ── Reading order: left col top→bottom, then right col top→bottom ─────
        left  = sorted([b for b in raw if b["column_idx"] == 0], key=lambda b: b["bbox"][1])
        right = sorted([b for b in raw if b["column_idx"] == 1], key=lambda b: b["bbox"][1])
        ordered = left + right

        return [
            LayoutBlock(
                label=b["label"],
                bbox=b["bbox"],
                crop=b["crop"],
                column_idx=b["column_idx"],
                reading_order=idx,
                confidence=round(b["conf"], 3),
            )
            for idx, b in enumerate(ordered)
        ]

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-initialise DocLayout-YOLO with the weights in LayoutAnalyser/."""
        if self._model is not None:
            return
        from doclayout_yolo import YOLOv10
        weights = str(_WEIGHTS_PT.resolve())
        self._model = YOLOv10(weights)

    # ── Column detection ───────────────────────────────────────────────────────

    def _detect_column_boundary(
        self, blocks: list[dict], img_w: int
    ) -> Optional[float]:
        """
        Return the x-coordinate of the column split, or None for single-column.

        Heuristic: if a significant number of blocks have their centres on the
        left (<35%) AND on the right (>65%) of the page, with few in the middle,
        the page is two-column and we split at the midpoint.
        """
        if len(blocks) < 4:
            return None

        x_mids = np.array([b["x_mid"] for b in blocks])
        left_count  = int(np.sum(x_mids < img_w * 0.38))
        right_count = int(np.sum(x_mids > img_w * 0.62))
        mid_count   = int(np.sum((x_mids >= img_w * 0.38) & (x_mids <= img_w * 0.62)))
        total = len(blocks)

        # Both sides must be substantially populated; the middle must be sparse
        if (
            left_count  >= max(2, total * 0.25)
            and right_count >= max(2, total * 0.25)
            and mid_count   <= max(1, total * 0.15)
        ):
            return float(img_w * 0.5)

        return None
