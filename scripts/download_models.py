#!/usr/bin/env python3
"""
Pre-download all ML model weights with visible progress.
Run once after pip install — before starting the server.

Usage:
    source ~/.venvs/extractor_ocr/bin/activate
    python scripts/download_models.py
"""
import sys
import os

# Make ml/ importable without Django
_ML_DIR = os.path.join(os.path.dirname(__file__), "../extractor/ml")
sys.path.insert(0, os.path.abspath(_ML_DIR))

def step(n, total, label):
    print(f"[{n}/{total}] {label}", flush=True)

def done():
    print("      done.", flush=True)


step(1, 3, "Layout detection model — DocLayout-YOLO (~39 MB)")
from layout import ExamLayoutParser
ExamLayoutParser()._load_model()
done()

step(2, 3, "OCR models — MFD + CnOCR + formula encoder/decoder (~260 MB)")
import ocr_pipeline
ocr_pipeline.init_models()
done()

step(3, 3, "Table recognition model — table-rec (~110 MB)")
try:
    from pix2text import TableOCR
    TableOCR.from_config()
    done()
except Exception as e:
    print(f"      warning: table model download failed ({e})")
    print("      Table OCR may fail until the model is available.")

print()
print("All models ready. You can now start the server.")
