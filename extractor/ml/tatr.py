"""
TATR — Table Structure Analysis
================================
Wraps microsoft/table-transformer-structure-recognition to detect
row / column structure within a table crop image.

Returns relative-fraction dividers (0.0–1.0) instead of absolute pixel
coordinates so that dividers scale correctly when the table bbox is
resized in the canvas.

Skew correction: estimates and corrects scan rotation (±10°) using the
projection-profile method before feeding TATR, so dividers align with
actual cell boundaries even on tilted scans. The corrected angle is
stored as `deskew_angle` so the OCR pipeline can apply the same rotation
when cropping individual cells.

Cell types: generates an initial (n_rows × n_cols) cell_types grid from
TATR structure — header rows get type "header", body rows get "data".
Users can override individual cells in the frontend.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_tatr_processor = None
_tatr_model = None

_LABEL_ROW           = "table row"
_LABEL_COL           = "table column"
_LABEL_COL_HEADER    = "table column header"
_DETECTION_THRESHOLD = 0.7

# Skew search range and step (degrees)
_SKEW_RANGE  = 10.0
_SKEW_STEP   = 0.5
# Skip deskew if the crop is too small (noisy projection profiles)
_DESKEW_MIN_DIM = 60


def init_tatr() -> None:
    """Pre-load TATR model. Call once at server startup."""
    _get_tatr()


def _get_tatr():
    global _tatr_processor, _tatr_model
    if _tatr_model is None:
        from transformers import TableTransformerForObjectDetection, DetrImageProcessor
        _tatr_processor = DetrImageProcessor.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
        _tatr_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition"
        )
    return _tatr_processor, _tatr_model


# ── Deskew helpers ────────────────────────────────────────────────────────────

def estimate_skew_angle(crop: Image.Image) -> float:
    """
    Estimate the rotation angle (degrees) of a table crop using the
    projection-profile method.

    Rotates a binary version of the crop through ±SKEW_RANGE degrees and
    returns the angle whose horizontal projection sums have maximum variance
    (sharp peaks = rows are horizontal = crop is aligned).

    Returns 0.0 when the crop is too small for a reliable estimate.
    """
    w, h = crop.size
    if w < _DESKEW_MIN_DIM or h < _DESKEW_MIN_DIM:
        return 0.0

    gray = np.array(crop.convert("L"), dtype=np.uint8)
    # Binary: dark pixels (text/lines) are 255, background is 0
    binary = np.where(gray < 160, np.uint8(255), np.uint8(0))

    angles = np.arange(-_SKEW_RANGE, _SKEW_RANGE + _SKEW_STEP / 2, _SKEW_STEP)
    best_angle = 0.0
    best_variance = -1.0

    for angle in angles:
        rotated_img = Image.fromarray(binary).rotate(
            angle, expand=False, fillcolor=0, resample=Image.BILINEAR
        )
        rotated = np.array(rotated_img)
        row_sums = rotated.sum(axis=1)
        variance = float(np.var(row_sums))
        if variance > best_variance:
            best_variance = variance
            best_angle = float(angle)

    return round(best_angle, 2)


def deskew_crop(crop: Image.Image, angle: float | None = None) -> tuple[Image.Image, float]:
    """
    Deskew a table crop.

    If angle is None, estimates it first with estimate_skew_angle().
    Returns (deskewed_crop, angle_applied). The angle is 0.0 when no
    correction was needed or the crop was too small.
    """
    if angle is None:
        angle = estimate_skew_angle(crop)

    if angle == 0.0:
        return crop, 0.0

    rotated = crop.rotate(angle, expand=False, fillcolor=(255, 255, 255),
                          resample=Image.BILINEAR)
    return rotated, angle


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_table_structure(crop: Image.Image, threshold: float = _DETECTION_THRESHOLD) -> dict:
    """
    Run TATR on a table crop and return relative-fraction dividers.

    Applies skew correction before running the model.  The correction angle
    is stored in ``deskew_angle`` so the OCR pipeline can apply the same
    rotation when cropping individual cells.

    Returns:
        {
            "row_dividers": [0.18, 0.37, ...],   # 0.0–1.0 fractions of crop height
            "col_dividers": [0.25, 0.51, ...],   # 0.0–1.0 fractions of crop width
            "header_rows":  1,
            "cell_types":   [["header", ...], ["data", ...], ...],
            "deskew_angle": 1.5,                 # degrees applied to correct skew
            "source":       "tatr"
        }
    """
    import torch

    # Deskew before feeding the model
    deskewed_crop, deskew_angle = deskew_crop(crop)

    processor, model = _get_tatr()
    crop_w, crop_h = deskewed_crop.size

    inputs = processor(images=deskewed_crop, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([deskewed_crop.size[::-1]])  # [H, W]
    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=target_sizes,
    )[0]

    id2label = model.config.id2label

    rows = []
    cols = []
    header_rows = 0

    for label_id, box in zip(results["labels"], results["boxes"]):
        label = id2label[label_id.item()]
        b = box.tolist()  # [x1, y1, x2, y2]

        if label == _LABEL_ROW:
            rows.append(b)
        elif label == _LABEL_COL:
            cols.append(b)
        elif label == _LABEL_COL_HEADER:
            header_rows += 1

    rows.sort(key=lambda b: b[1])   # sort by y1
    cols.sort(key=lambda b: b[0])   # sort by x1

    row_dividers = _boxes_to_dividers(rows, trailing_idx=3, leading_idx=1, dim=crop_h)
    col_dividers = _boxes_to_dividers(cols, trailing_idx=2, leading_idx=0, dim=crop_w)

    n_rows = len(row_dividers) + 1
    n_cols = len(col_dividers) + 1
    cell_types = _make_cell_types(n_rows, n_cols, header_rows)

    return {
        "row_dividers": row_dividers,
        "col_dividers": col_dividers,
        "header_rows":  header_rows,
        "cell_types":   cell_types,
        "deskew_angle": deskew_angle,
        "source":       "tatr",
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _boxes_to_dividers(
    boxes: list,
    trailing_idx: int,
    leading_idx: int,
    dim: int,
) -> list:
    """
    Convert sorted bounding boxes into relative-fraction divider positions.

    For N boxes → N-1 dividers. Each divider is the midpoint between the
    trailing edge of box[i] and the leading edge of box[i+1], normalized
    by dim.
    """
    if len(boxes) <= 1:
        return []

    dividers = []
    for i in range(len(boxes) - 1):
        midpoint = (boxes[i][trailing_idx] + boxes[i + 1][leading_idx]) / 2.0
        dividers.append(round(midpoint / dim, 4))

    return dividers


def _make_cell_types(n_rows: int, n_cols: int, header_rows: int) -> list[list[str]]:
    """
    Build an (n_rows × n_cols) grid of initial cell type strings.
    Rows within header_rows get "header"; all others get "data".
    """
    result = []
    for r in range(n_rows):
        cell_type = "header" if r < header_rows else "data"
        result.append([cell_type] * n_cols)
    return result
