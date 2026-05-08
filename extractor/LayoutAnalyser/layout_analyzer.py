"""
layout_analyzer.py
──────────────────
FastAPI endpoint that runs DocLayout-YOLO on a page image
and returns all detected content blocks with type, coordinates,
confidence, and extracted content per block.

Add to your existing main.py or run standalone:
    uvicorn layout_analyzer:app --reload --port 8001
"""

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

app = FastAPI(title="Layout Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DocLayout-YOLO class labels ──────────────────────────────────
# These are the exact classes the model was trained on
LABELS = {
    0:  "title",
    1:  "plain_text",
    2:  "abandon",          # headers, footers, page numbers — noise
    3:  "figure",
    4:  "figure_caption",
    5:  "table",
    6:  "table_caption",
    7:  "table_footnote",
    8:  "isolate_formula",  # display/block math equation
    9:  "formula_caption",
}

# Human-readable descriptions for the viewer
LABEL_META = {
    "title":            {"color": "#f59e0b", "desc": "Section or chapter heading"},
    "plain_text":       {"color": "#60a5fa", "desc": "Regular paragraph text"},
    "abandon":          {"color": "#6b7280", "desc": "Page noise — header/footer/page number"},
    "figure":           {"color": "#34d399", "desc": "Diagram, graph, or illustration"},
    "figure_caption":   {"color": "#6ee7b7", "desc": "Caption below a figure"},
    "table":            {"color": "#f472b6", "desc": "Data or comparison table"},
    "table_caption":    {"color": "#f9a8d4", "desc": "Caption above/below a table"},
    "table_footnote":   {"color": "#e879f9", "desc": "Footnote below a table"},
    "isolate_formula":  {"color": "#fb923c", "desc": "Display math / block equation"},
    "formula_caption":  {"color": "#fed7aa", "desc": "Label or number beside a formula"},
}

# ── Model singleton ───────────────────────────────────────────────
_layout_model = None

def get_layout_model():
    global _layout_model
    if _layout_model is None:
        from doclayout_yolo import YOLOv10
        # uses the bundled weights from pix2text's installation
        import doclayout_yolo
        import os
        # find the model weights path
        pkg_dir = os.path.dirname(doclayout_yolo.__file__)
        weight_candidates = [
            os.path.join(pkg_dir, "weights", "weights.pt"),
            os.path.join(pkg_dir, "DocLayout-YOLO-DocStructBench.pt"),
        ]
        weight_path = None
        for c in weight_candidates:
            if os.path.exists(c):
                weight_path = c
                break

        if weight_path is None:
            # fallback: let doclayout_yolo find it automatically
            _layout_model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
        else:
            _layout_model = YOLOv10(weight_path)

    return _layout_model


# ── Health ────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main endpoint ─────────────────────────────────────────────────
@app.post("/layout")
async def analyze_layout(file: UploadFile = File(...)):
    """
    Upload a page image → returns detected content blocks.

    Each block contains:
      - id, label, description, color
      - bbox: [x1, y1, x2, y2] in pixels
      - bbox_pct: [x1, y1, x2, y2] as % of image size (for frontend overlay)
      - confidence
      - area_pct: how much of the page this block covers
      - reading_order: approximate top-to-bottom, left-to-right order
    """
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_w, img_h = img.size

    model = get_layout_model()

    # Run inference
    results = model.predict(
        img,
        imgsz=1024,
        conf=0.25,      # confidence threshold — lower = more detections
        iou=0.45,       # NMS threshold
        verbose=False,
    )

    blocks = []
    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        for i, box in enumerate(boxes):
            cls_id   = int(box.cls[0].item())
            conf     = round(float(box.conf[0].item()), 3)
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

            label = LABELS.get(cls_id, "unknown")
            meta  = LABEL_META.get(label, {"color": "#ffffff", "desc": ""})

            area   = (x2 - x1) * (y2 - y1)
            img_area = img_w * img_h

            blocks.append({
                "id":           i,
                "label":        label,
                "description":  meta["desc"],
                "color":        meta["color"],
                "confidence":   conf,
                "bbox":         [round(x1), round(y1), round(x2), round(y2)],
                "bbox_pct": {
                    "x":  round(x1 / img_w * 100, 2),
                    "y":  round(y1 / img_h * 100, 2),
                    "w":  round((x2 - x1) / img_w * 100, 2),
                    "h":  round((y2 - y1) / img_h * 100, 2),
                },
                "area_pct":     round(area / img_area * 100, 2),
            })

    # Sort by reading order: top → bottom, left → right
    blocks.sort(key=lambda b: (b["bbox"][1] // 50, b["bbox"][0]))
    for i, b in enumerate(blocks):
        b["reading_order"] = i + 1

    # Summary stats
    summary = {}
    for b in blocks:
        lbl = b["label"]
        summary[lbl] = summary.get(lbl, 0) + 1

    return {
        "image_size":   {"width": img_w, "height": img_h},
        "total_blocks": len(blocks),
        "summary":      summary,
        "label_meta":   LABEL_META, 
        "blocks":       blocks,
    }