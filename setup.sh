#!/usr/bin/env bash
# Run from the repo root (ContentExtractorBackend/).
set -euo pipefail

echo "=== Content Extractor Backend — first-time setup ==="
echo ""

# ── 1. Python venv ─────────────────────────────────────────────────────────────
echo "[1/3] Creating Python virtual environment..."
python3.11 -m venv ~/.venvs/extractor_ocr
source ~/.venvs/extractor_ocr/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt
echo "      done."
echo ""

# ── 2. Django migrations ────────────────────────────────────────────────────────
echo "[2/3] Running database migrations..."
python manage.py migrate --run-syncdb
echo "      done."
echo ""

# ── 3. ML model download ────────────────────────────────────────────────────────
echo "[3/3] Downloading ML models (~400 MB, one-time)..."
python scripts/download_models.py
echo ""

echo "=== Setup complete ==="
echo ""
echo "To start the backend:"
echo "  source ~/.venvs/extractor_ocr/bin/activate"
echo "  python manage.py runserver 8001"
