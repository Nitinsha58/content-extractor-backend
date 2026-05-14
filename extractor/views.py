"""
API views for the extractor app.
"""

import io
import uuid
import time
import zipfile
from django.conf import settings
from django.http import HttpResponse
from django.db.models import Q, Sum, Count
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image

from . import ml_state


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_restore_session(session_id: str) -> dict | None:
    """
    Return the session dict for session_id. If it's missing from the in-memory
    cache (e.g. after a server restart), try to restore it from the disk cache
    and training images directories.
    """
    session = ml_state._debug_sessions.get(session_id)
    if session:
        return session

    # Try to restore from disk
    for cache_dir in (settings.DEBUG_CACHE_DIR, settings.TRAINING_IMAGES_DIR):
        candidate = cache_dir / f"{session_id}.jpg"
        if candidate.exists():
            try:
                img = Image.open(candidate)
                w, h = img.size
                img.close()
            except Exception:
                continue
            restored = {"image_path": candidate, "image_w": w, "image_h": h}
            ml_state._debug_sessions[session_id] = restored
            return restored

    return None

_IMAGE_EXTS = {'jpg', 'jpeg', 'png', 'tiff', 'tif', 'bmp', 'webp', 'gif'}

def _get_file_type(filename):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext == 'pdf':
        return 'pdf'
    if ext in _IMAGE_EXTS:
        return 'image'
    return 'other'

def _log_activity(document, action):
    from .models import Activity
    Activity.objects.create(
        document=document,
        doc_label=document.filename,
        action=action,
    )

def _delete_document_files(doc, *, include_originals: bool):
    """
    Delete files associated with a document.

    include_originals=False  — soft-delete path: remove processing artifacts
        (.debug_cache session images, training_images copies, tatr_crops, media/figures)
        but keep pdf_file + thumbnail so the document can still be restored from Trash.

    include_originals=True   — hard-delete path: everything above plus pdf_file + thumbnail.

    Must be called BEFORE doc.delete() because the page rows are still needed to
    collect session IDs and figure URLs.
    """
    import glob as _glob
    import json as _json
    import pathlib as _pl
    import re as _re

    _FIG_RE = _re.compile(r'/media/figures/[^"\'<>\s]+')

    # Collect per-page artefacts from the DB before the CASCADE wipes them.
    session_ids = list(doc.pages.values_list('session_id', flat=True))
    figure_paths: set[_pl.Path] = set()
    for page_vals in doc.pages.values('ocr_blocks', 'structured_content'):
        for field_val in (page_vals['ocr_blocks'], page_vals['structured_content']):
            if not field_val:
                continue
            raw = field_val if isinstance(field_val, str) else _json.dumps(field_val)
            for url in _FIG_RE.findall(raw):
                figure_paths.add(settings.MEDIA_ROOT / url.removeprefix('/media/'))

    # Session images (.debug_cache + training_images) and TATR crops
    for sid in session_ids:
        if not sid:
            continue
        for path in (
            settings.DEBUG_CACHE_DIR / f"{sid}.jpg",
            settings.TRAINING_IMAGES_DIR / f"{sid}.jpg",
        ):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        for crop in _glob.glob(str(settings.TATR_CROPS_DIR / f"{sid}_block_*.jpg")):
            try:
                _pl.Path(crop).unlink(missing_ok=True)
            except Exception:
                pass

    # Figure crops (media/figures/)
    for path in figure_paths:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    # Original uploaded file + thumbnail — only on hard delete
    if include_originals:
        if doc.pdf_file:
            doc.pdf_file.delete(save=False)
        if doc.thumbnail:
            doc.thumbnail.delete(save=False)


def _doc_to_dict(doc, request):
    pages = list(doc.pages.all())
    return {
        'id': str(doc.id),
        'filename': doc.filename,
        'page_count': doc.page_count,
        'status': doc.status,
        'thumbnail_url': request.build_absolute_uri(doc.thumbnail.url) if doc.thumbnail else None,
        'pdf_url': request.build_absolute_uri(doc.pdf_file.url),
        'created_at': doc.created_at.isoformat(),
        'updated_at': doc.updated_at.isoformat(),
        'is_starred': doc.is_starred,
        'folder_id': str(doc.folder_id) if doc.folder_id else None,
        'tag_ids': [str(t.id) for t in doc.tags.all()],
        'file_size': doc.file_size,
        'file_type': doc.file_type,
        'deleted_at': doc.deleted_at.isoformat() if doc.deleted_at else None,
        'pages_layout_done': sum(1 for p in pages if p.status == 'layout-detected'),
        'pages_ocr_done': sum(1 for p in pages if p.status == 'ocr-complete'),
    }


# ── ML processing views ───────────────────────────────────────────────────────

class LayoutView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        file = request.FILES.get("file")
        if not file:
            return Response({"error": "file required"}, status=400)

        try:
            img = Image.open(file).convert("RGB")
        except Exception as e:
            return Response({"error": f"Invalid image: {e}"}, status=400)

        session_id = str(uuid.uuid4())
        cache_dir = settings.DEBUG_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{session_id}.jpg"

        try:
            img.save(cache_path, "JPEG", quality=85)
        except Exception as e:
            return Response({"error": f"Failed to save image: {e}"}, status=500)

        try:
            training_path = settings.TRAINING_IMAGES_DIR / f"{session_id}.jpg"
            img.save(training_path, "JPEG", quality=85)
        except Exception:
            pass  # non-fatal — detection can still proceed without a training image

        ml_state._debug_sessions[session_id] = {
            "image_path": cache_path,
            "image_w": img.width,
            "image_h": img.height,
        }

        try:
            layout_parser = ml_state.get_layout_parser()
            layout_blocks = layout_parser.parse(img)
        except Exception as e:
            ml_state.clear_session(session_id)
            return Response({"error": f"Layout detection failed: {e}"}, status=500)

        blocks_data = [
            {
                "id": str(uuid.uuid4()),
                "label": lb.label,
                "bbox": [int(x) for x in lb.bbox],
                "confidence": float(lb.confidence),
                "column_idx": lb.column_idx,
                "reading_order": lb.reading_order,
            }
            for lb in layout_blocks
        ]

        return Response({
            "session_id": session_id,
            "image_url": f"/api/debug/session-image/{session_id}/",
            "image_width": img.width,
            "image_height": img.height,
            "layout_blocks": blocks_data,
        })


class OcrView(APIView):

    def post(self, request):
        session_id = request.data.get("session_id")
        layout_blocks_data = request.data.get("layout_blocks", [])

        if not session_id:
            return Response({"error": "session_id required"}, status=400)

        session = _get_or_restore_session(session_id)
        if not session:
            return Response({"error": "session not found"}, status=404)

        try:
            img = Image.open(session["image_path"]).convert("RGB")
        except Exception as e:
            return Response({"error": f"Failed to load cached image: {e}"}, status=500)

        try:
            from .ml.layout import LayoutBlock
            from .ml.ocr_pipeline import process_region
        except ImportError as e:
            return Response({"error": f"ML modules not available: {e}"}, status=500)

        ocr_results = []
        for lb_data in layout_blocks_data:
            try:
                x1, y1, x2, y2 = lb_data["bbox"]
                # Expand bbox slightly before cropping to recover content that
                # sits right at the detection boundary (superscripts, descenders,
                # border strokes). Skip for table: cell dividers are proportional
                # to the original crop dimensions and expansion would misalign them.
                if lb_data.get("label") != "table":
                    bw, bh = x2 - x1, y2 - y1
                    expand = max(4, int(min(bw, bh) * 0.02))
                    x1 = max(0, x1 - expand)
                    y1 = max(0, y1 - expand)
                    x2 = min(img.width, x2 + expand)
                    y2 = min(img.height, y2 + expand)
                crop = img.crop((x1, y1, x2, y2))
                lb = LayoutBlock(
                    label=lb_data["label"],
                    bbox=lb_data["bbox"],
                    crop=crop,
                    column_idx=lb_data.get("column_idx", 0),
                    reading_order=lb_data.get("reading_order", 0),
                    confidence=lb_data.get("confidence", 1.0),
                    table_structure=lb_data.get("table_structure"),
                )
                t0 = time.time()
                blocks = process_region(lb)
                duration = (time.time() - t0) * 1000
                ocr_results.append({
                    "block_id": lb_data.get("id", str(uuid.uuid4())),
                    "label": lb_data["label"],
                    "bbox": lb_data["bbox"],
                    "reading_order": lb_data.get("reading_order", 0),
                    "column_idx": lb_data.get("column_idx", 0),
                    "blocks": [b.model_dump() for b in blocks],
                    "duration_ms": round(duration, 1),
                })
            except Exception as e:
                print(f"OCR failed for block {lb_data.get('id', '?')}: {e}")
                ocr_results.append({
                    "block_id": lb_data.get("id", str(uuid.uuid4())),
                    "label": lb_data["label"],
                    "bbox": lb_data["bbox"],
                    "reading_order": lb_data.get("reading_order", 0),
                    "column_idx": lb_data.get("column_idx", 0),
                    "blocks": [],
                    "duration_ms": 0,
                    "error": str(e),
                })

        return Response({
            "session_id": session_id,
            "image_url": f"/api/debug/session-image/{session_id}/",
            "image_width": session["image_w"],
            "image_height": session["image_h"],
            "ocr_blocks": ocr_results,
        })


class CropView(APIView):

    def get(self, request):
        session_id = request.query_params.get("session_id")
        bbox_str = request.query_params.get("bbox", "")

        if not session_id:
            return Response({"error": "session_id required"}, status=400)

        session = _get_or_restore_session(session_id)
        if not session:
            return Response({"error": "session not found"}, status=404)

        if not bbox_str:
            return Response({"error": "bbox required (format: x1,y1,x2,y2)"}, status=400)

        try:
            bbox = [int(v) for v in bbox_str.split(",")]
            if len(bbox) != 4:
                raise ValueError("bbox must have 4 values")
            x1, y1, x2, y2 = bbox
        except Exception as e:
            return Response({"error": f"Invalid bbox: {e}"}, status=400)

        try:
            img = Image.open(session["image_path"]).convert("RGB")
            crop = img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, "PNG")
            buf.seek(0)
            return HttpResponse(buf.getvalue(), content_type="image/png")
        except Exception as e:
            return Response({"error": f"Failed to crop image: {e}"}, status=500)


class TableCellTypesView(APIView):
    """
    POST /api/debug/table-cell-types/

    Crops each cell defined by table_structure row/col dividers, runs
    DocLayout-YOLO on each crop, and returns a 2-D cell_types array using
    the same vocab as Canvas Block OCR labels: plain_text / title /
    isolate_formula / figure.  Defaults to plain_text when YOLO finds nothing.
    """

    _CELL_MIN_DIM = 128   # upscale to this before YOLO so small crops are legible

    def post(self, request):
        session_id      = request.data.get("session_id")
        block_id        = request.data.get("block_id")
        bbox            = request.data.get("bbox")
        table_structure = request.data.get("table_structure")

        if not session_id:
            return Response({"error": "session_id required"}, status=400)
        if not bbox or len(bbox) != 4:
            return Response({"error": "bbox required (4 values)"}, status=400)
        if not table_structure:
            return Response({"error": "table_structure required"}, status=400)

        session = _get_or_restore_session(session_id)
        if not session:
            return Response({"error": "session not found"}, status=404)

        try:
            img = Image.open(session["image_path"]).convert("RGB")
        except Exception as e:
            return Response({"error": f"Failed to load image: {e}"}, status=500)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        table_crop = img.crop((x1, y1, x2, y2))

        row_dividers = table_structure.get("row_dividers") or []
        col_dividers = table_structure.get("col_dividers") or []
        crop_w, crop_h = table_crop.size

        row_boundaries = [0] + [round(d * crop_h) for d in row_dividers] + [crop_h]
        col_boundaries = [0] + [round(d * crop_w) for d in col_dividers] + [crop_w]
        n_rows = len(row_boundaries) - 1
        n_cols = len(col_boundaries) - 1

        if n_rows == 0 or n_cols == 0:
            return Response({"error": "table_structure contains no rows or columns"}, status=400)

        try:
            from .ml.layout import ExamLayoutParser
            parser = ExamLayoutParser()
        except Exception as e:
            return Response({"error": f"Failed to load layout parser: {e}"}, status=500)

        cell_types = []
        for r in range(n_rows):
            row_types = []
            for c in range(n_cols):
                cell_crop = table_crop.crop((
                    col_boundaries[c], row_boundaries[r],
                    col_boundaries[c + 1], row_boundaries[r + 1],
                )).convert("RGB")

                cw, ch = cell_crop.size
                if cw < self._CELL_MIN_DIM or ch < self._CELL_MIN_DIM:
                    scale = max(self._CELL_MIN_DIM / max(cw, 1), self._CELL_MIN_DIM / max(ch, 1))
                    cell_crop = cell_crop.resize(
                        (max(1, round(cw * scale)), max(1, round(ch * scale))),
                        Image.LANCZOS,
                    )

                try:
                    blocks = parser.parse(cell_crop)
                    if blocks:
                        best = max(blocks, key=lambda b: b.confidence)
                        row_types.append(best.label)
                    else:
                        row_types.append("plain_text")
                except Exception:
                    row_types.append("plain_text")

            cell_types.append(row_types)

        return Response({"cell_types": cell_types})


class TableStructureView(APIView):

    def post(self, request):
        session_id = request.data.get("session_id")
        block_id   = request.data.get("block_id")
        bbox       = request.data.get("bbox")

        if not session_id:
            return Response({"error": "session_id required"}, status=400)
        if not bbox or len(bbox) != 4:
            return Response({"error": "bbox required (4 values: [x1, y1, x2, y2])"}, status=400)

        session = _get_or_restore_session(session_id)
        if not session:
            return Response({"error": "session not found"}, status=404)

        try:
            img = Image.open(session["image_path"]).convert("RGB")
        except Exception as e:
            return Response({"error": f"Failed to load image: {e}"}, status=500)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = img.crop((x1, y1, x2, y2))

        # Save crop to disk for COCO training data collection (Slice 05)
        crops_dir = settings.TATR_CROPS_DIR
        crop_filename = f"{session_id}_block_{block_id}.jpg"
        try:
            crop.save(crops_dir / crop_filename, "JPEG")
        except Exception:
            pass  # crop save failure is non-fatal

        try:
            from .ml.tatr import analyze_table_structure
            table_structure = analyze_table_structure(crop)
        except Exception as e:
            return Response({"error": f"Table structure analysis failed: {e}"}, status=500)

        return Response({"table_structure": table_structure})


class SessionImageView(APIView):

    def get(self, request, session_id):
        session = _get_or_restore_session(session_id)
        if not session:
            return Response({"error": "session not found"}, status=404)

        try:
            with open(session["image_path"], "rb") as f:
                return HttpResponse(f.read(), content_type="image/jpeg")
        except Exception as e:
            return Response({"error": f"Failed to serve image: {e}"}, status=500)


class ExportView(APIView):

    def post(self, request):
        session_id = request.data.get("session_id")
        fmt = request.data.get("format", "html")
        ocr_blocks_data = request.data.get("ocr_blocks", [])

        if not session_id:
            return Response({"error": "session_id required"}, status=400)

        if fmt not in ("html", "markdown", "docx"):
            return Response({"error": f"Invalid format: {fmt}"}, status=400)

        session = _get_or_restore_session(session_id)
        if not session:
            return Response({"error": "session not found"}, status=404)

        try:
            from .ml.schema import DebugOCRBlock
            from .ml.document_builder import build_document_tree
        except ImportError as e:
            return Response({"error": f"ML modules not available: {e}"}, status=500)

        try:
            ocr_blocks = [DebugOCRBlock(**b) for b in ocr_blocks_data]
            page_node = build_document_tree(
                ocr_blocks=ocr_blocks,
                img_w=session["image_w"],
                img_h=session["image_h"],
                session_id=session_id,
                image_url=f"/api/debug/session-image/{session_id}/",
            )

            if fmt == "html":
                from .ml.exporters.html_exporter import export_html
                content = export_html(page_node, base_url="http://localhost:8001/")
                return HttpResponse(content, content_type="text/html",
                    headers={"Content-Disposition": 'attachment; filename="document.html"'})

            elif fmt == "markdown":
                from .ml.exporters.markdown_exporter import export_markdown
                content = export_markdown(page_node)
                return HttpResponse(content, content_type="text/markdown",
                    headers={"Content-Disposition": 'attachment; filename="document.md"'})

            elif fmt == "docx":
                from .ml.exporters.docx_exporter import export_docx
                content = export_docx(page_node)
                return HttpResponse(bytes(content),
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={"Content-Disposition": 'attachment; filename="document.docx"'})

        except Exception as e:
            return Response({"error": f"Export failed: {e}"}, status=500)


# ── Document CRUD ─────────────────────────────────────────────────────────────

class DocumentListCreateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request):
        from .models import Document
        qs = Document.objects.prefetch_related('tags', 'pages')

        # deleted filter (default: exclude deleted)
        if request.query_params.get('deleted') == '1':
            qs = qs.filter(deleted_at__isnull=False)
        else:
            qs = qs.filter(deleted_at__isnull=True)

        # folder filter
        folder = request.query_params.get('folder')
        if folder:
            qs = qs.filter(folder_id=folder)

        # starred filter
        if request.query_params.get('starred') == '1':
            qs = qs.filter(is_starred=True)

        # tag filter
        tag = request.query_params.get('tag')
        if tag:
            qs = qs.filter(tags__id=tag)

        # server-side filename search
        q = request.query_params.get('q', '').strip()
        if q:
            qs = qs.filter(filename__icontains=q)

        return Response([_doc_to_dict(doc, request) for doc in qs])

    def post(self, request):
        from .models import Document
        pdf_file = request.FILES.get('pdf')
        if not pdf_file:
            return Response({'error': 'pdf file required'}, status=400)

        filename = request.data.get('filename') or pdf_file.name
        page_count = int(request.data.get('page_count', 0))
        file_size = pdf_file.size
        file_type = _get_file_type(filename)

        folder = None
        folder_id = request.data.get('folder_id')
        if folder_id:
            try:
                from .models import Folder
                folder = Folder.objects.get(id=folder_id)
            except Exception:
                pass

        doc = Document.objects.create(
            filename=filename,
            pdf_file=pdf_file,
            page_count=page_count,
            status='uploaded',
            file_size=file_size,
            file_type=file_type,
            folder=folder,
        )

        # Generate thumbnail
        from django.core.files.base import ContentFile
        thumb_bytes = None
        try:
            doc.pdf_file.seek(0)
            img = Image.open(doc.pdf_file).convert('RGB')
            img.thumbnail((400, 600))
            buf = io.BytesIO()
            img.save(buf, 'JPEG', quality=85)
            thumb_bytes = buf.getvalue()
        except Exception:
            try:
                import fitz
                doc.pdf_file.seek(0)
                pdf_bytes = doc.pdf_file.read()
                pdf_doc = fitz.open(stream=pdf_bytes, filetype='pdf')
                pix = pdf_doc[0].get_pixmap(matrix=fitz.Matrix(0.5, 0.5))
                thumb_bytes = pix.tobytes('jpeg')
                pdf_doc.close()
            except Exception as e:
                print(f'Thumbnail generation failed: {e}')
        if thumb_bytes:
            doc.thumbnail.save(f'{doc.id}_thumb.jpg', ContentFile(thumb_bytes), save=True)

        _log_activity(doc, 'uploaded')

        return Response({
            'id': str(doc.id),
            'filename': doc.filename,
            'page_count': doc.page_count,
            'status': doc.status,
            'thumbnail_url': request.build_absolute_uri(doc.thumbnail.url) if doc.thumbnail else None,
            'pdf_url': request.build_absolute_uri(doc.pdf_file.url),
            'created_at': doc.created_at.isoformat(),
            'file_size': doc.file_size,
            'file_type': doc.file_type,
        }, status=201)


class DocumentDetailView(APIView):

    def get(self, request, doc_id):
        from .models import Document
        try:
            doc = Document.objects.prefetch_related('tags', 'pages').get(id=doc_id, deleted_at__isnull=True)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)

        pages_data = [{
            'id': str(p.id),
            'page_number': p.page_number,
            'session_id': p.session_id,
            'image_w': p.image_w,
            'image_h': p.image_h,
            'layout_blocks': p.layout_blocks,
            'ocr_blocks': p.ocr_blocks,
            'status': p.status,
        } for p in doc.pages.all()]

        return Response({
            'id': str(doc.id),
            'filename': doc.filename,
            'page_count': doc.page_count,
            'status': doc.status,
            'thumbnail_url': request.build_absolute_uri(doc.thumbnail.url) if doc.thumbnail else None,
            'pdf_url': request.build_absolute_uri(doc.pdf_file.url),
            'created_at': doc.created_at.isoformat(),
            'is_starred': doc.is_starred,
            'folder_id': str(doc.folder_id) if doc.folder_id else None,
            'tag_ids': [str(t.id) for t in doc.tags.all()],
            'pages': pages_data,
        })

    def patch(self, request, doc_id):
        from .models import Document
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)

        if 'page_count' in request.data:
            doc.page_count = int(request.data['page_count'])
            doc.save(update_fields=['page_count', 'updated_at'])

        return Response({'id': str(doc.id), 'page_count': doc.page_count})

    def delete(self, request, doc_id):
        from .models import Document
        from django.utils import timezone
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)

        if request.query_params.get('hard') == '1':
            # Hard delete: clean up processing artifacts + original files, then remove record.
            _delete_document_files(doc, include_originals=True)
            doc.delete()
        else:
            # Soft delete: move to Trash.
            # Processing artifacts (session images, figure crops, TATR crops) are cleaned
            # up immediately — they are large and not needed to restore the document.
            # The original uploaded file and thumbnail are kept so Restore still works.
            _delete_document_files(doc, include_originals=False)
            doc.deleted_at = timezone.now()
            doc.save(update_fields=['deleted_at', 'updated_at'])

        return Response(status=204)


class PageSaveView(APIView):

    def get(self, request, doc_id, page_number):
        from .models import Document, DocumentPage
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'document not found'}, status=404)
        try:
            page = DocumentPage.objects.get(document=doc, page_number=page_number)
            return Response({
                'page_number': page.page_number,
                'session_id': page.session_id,
                'image_w': page.image_w,
                'image_h': page.image_h,
                'layout_blocks': page.layout_blocks or [],
                'ocr_blocks': page.ocr_blocks or [],
                'status': page.status,
            })
        except DocumentPage.DoesNotExist:
            return Response({'layout_blocks': [], 'status': 'idle'})

    def post(self, request, doc_id, page_number):
        from .models import Document, DocumentPage
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'document not found'}, status=404)

        page, _ = DocumentPage.objects.get_or_create(document=doc, page_number=page_number)

        fields = []
        for field in ('session_id', 'image_w', 'image_h', 'layout_blocks', 'ocr_blocks',
                      'status', 'structured_content', 'structure_status'):
            if field in request.data:
                val = request.data[field]
                if field in ('image_w', 'image_h'):
                    val = int(val)
                setattr(page, field, val)
                fields.append(field)

        # Invalidate cached structured content whenever ocr_blocks are updated
        if 'ocr_blocks' in fields and 'structured_content' not in fields:
            page.structured_content = None
            page.structure_status = 'none'
            fields += ['structured_content', 'structure_status']

        if fields:
            page.save(update_fields=fields)

        # Recompute document-level status
        all_pages = list(doc.pages.all())
        ocr_done = sum(1 for p in all_pages if p.status == 'ocr-complete')
        layout_done = sum(1 for p in all_pages if p.status in ('layout-detected', 'ocr-complete'))

        was_complete = doc.status == 'complete'
        if doc.page_count > 0 and ocr_done == doc.page_count:
            doc.status = 'complete'
        elif layout_done > 0:
            doc.status = 'partial'
        doc.save(update_fields=['status', 'updated_at'])

        if not was_complete and doc.status == 'complete':
            _log_activity(doc, 'ocr_completed')

        return Response({'id': str(page.id), 'page_number': page.page_number, 'status': page.status})


# ── Document actions ──────────────────────────────────────────────────────────

class DocumentStarView(APIView):

    def post(self, request, doc_id):
        from .models import Document
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)
        doc.is_starred = not doc.is_starred
        doc.save(update_fields=['is_starred', 'updated_at'])
        return Response({'is_starred': doc.is_starred})


class DocumentMoveView(APIView):

    def post(self, request, doc_id):
        from .models import Document, Folder
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)

        folder_id = request.data.get('folder_id')
        if folder_id:
            try:
                doc.folder = Folder.objects.get(id=folder_id)
            except Folder.DoesNotExist:
                return Response({'error': 'folder not found'}, status=404)
        else:
            doc.folder = None
        doc.save(update_fields=['folder', 'updated_at'])
        return Response({'folder_id': str(doc.folder_id) if doc.folder_id else None})


class DocumentTagsView(APIView):

    def put(self, request, doc_id):
        from .models import Document
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)
        doc.tags.set(request.data.get('tag_ids', []))
        return Response({'tag_ids': [str(t.id) for t in doc.tags.all()]})


class DocumentRestoreView(APIView):

    def post(self, request, doc_id):
        from .models import Document
        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'not found'}, status=404)
        doc.deleted_at = None
        doc.save(update_fields=['deleted_at', 'updated_at'])
        return Response({'status': 'restored'})


# ── Folders ───────────────────────────────────────────────────────────────────

class FolderListCreateView(APIView):

    def get(self, request):
        from .models import Folder
        folders = Folder.objects.annotate(
            doc_count=Count('documents', filter=Q(documents__deleted_at__isnull=True))
        )
        return Response([{
            'id': str(f.id),
            'name': f.name,
            'parent_id': str(f.parent_id) if f.parent_id else None,
            'doc_count': f.doc_count,
            'created_at': f.created_at.isoformat(),
        } for f in folders])

    def post(self, request):
        from .models import Folder
        name = (request.data.get('name') or '').strip()
        if not name:
            return Response({'error': 'name required'}, status=400)

        parent = None
        parent_id = request.data.get('parent_id')
        if parent_id:
            try:
                parent = Folder.objects.get(id=parent_id)
            except Folder.DoesNotExist:
                return Response({'error': 'parent not found'}, status=404)

        folder = Folder.objects.create(name=name, parent=parent)
        return Response({
            'id': str(folder.id),
            'name': folder.name,
            'parent_id': str(folder.parent_id) if folder.parent_id else None,
            'doc_count': 0,
        }, status=201)


class FolderDetailView(APIView):

    def patch(self, request, folder_id):
        from .models import Folder
        try:
            folder = Folder.objects.get(id=folder_id)
        except Folder.DoesNotExist:
            return Response({'error': 'not found'}, status=404)

        if 'name' in request.data:
            folder.name = (request.data['name'] or '').strip()
        if 'parent_id' in request.data:
            parent_id = request.data['parent_id']
            if parent_id:
                try:
                    folder.parent = Folder.objects.get(id=parent_id)
                except Folder.DoesNotExist:
                    return Response({'error': 'parent not found'}, status=404)
            else:
                folder.parent = None
        folder.save()
        return Response({
            'id': str(folder.id),
            'name': folder.name,
            'parent_id': str(folder.parent_id) if folder.parent_id else None,
        })

    def delete(self, request, folder_id):
        from .models import Folder
        try:
            folder = Folder.objects.get(id=folder_id)
        except Folder.DoesNotExist:
            return Response({'error': 'not found'}, status=404)
        folder.delete()  # CASCADE: deletes child folders; SET_NULL: docs move to root
        return Response(status=204)


# ── Tags ──────────────────────────────────────────────────────────────────────

class TagListCreateView(APIView):

    def get(self, request):
        from .models import Tag
        tags = Tag.objects.annotate(
            doc_count=Count('documents', filter=Q(documents__deleted_at__isnull=True))
        )
        return Response([{
            'id': str(t.id),
            'name': t.name,
            'color': t.color,
            'doc_count': t.doc_count,
        } for t in tags])

    def post(self, request):
        from .models import Tag
        name = (request.data.get('name') or '').strip()
        if not name:
            return Response({'error': 'name required'}, status=400)
        color = request.data.get('color', 'gray')
        if Tag.objects.filter(name=name).exists():
            return Response({'error': 'tag already exists'}, status=409)
        tag = Tag.objects.create(name=name, color=color)
        return Response({'id': str(tag.id), 'name': tag.name, 'color': tag.color, 'doc_count': 0}, status=201)


class TagDetailView(APIView):

    def patch(self, request, tag_id):
        from .models import Tag
        try:
            tag = Tag.objects.get(id=tag_id)
        except Tag.DoesNotExist:
            return Response({'error': 'not found'}, status=404)
        if 'name' in request.data:
            tag.name = (request.data['name'] or '').strip()
        if 'color' in request.data:
            tag.color = request.data['color']
        tag.save()
        return Response({'id': str(tag.id), 'name': tag.name, 'color': tag.color})

    def delete(self, request, tag_id):
        from .models import Tag
        try:
            tag = Tag.objects.get(id=tag_id)
        except Tag.DoesNotExist:
            return Response({'error': 'not found'}, status=404)
        tag.delete()
        return Response(status=204)


# ── Dashboard data ────────────────────────────────────────────────────────────

class ActivityListView(APIView):

    def get(self, request):
        from .models import Activity
        limit = min(int(request.query_params.get('limit', 10)), 50)
        activities = Activity.objects.all()[:limit]
        return Response([{
            'id': str(a.id),
            'doc_id': str(a.document_id) if a.document_id else None,
            'doc_label': a.doc_label,
            'action': a.action,
            'created_at': a.created_at.isoformat(),
        } for a in activities])


class StorageView(APIView):

    def get(self, request):
        from .models import Document, DocumentPage
        docs = Document.objects.filter(deleted_at__isnull=True)
        file_count = docs.count()
        used_bytes = docs.aggregate(total=Sum('file_size'))['total'] or 0

        type_counts = list(docs.values('file_type').annotate(count=Count('id')))
        by_type = [{
            'type': tc['file_type'],
            'count': tc['count'],
            'percent': round(tc['count'] / file_count * 100) if file_count > 0 else 0,
        } for tc in type_counts]

        ocr_pages = DocumentPage.objects.filter(
            document__deleted_at__isnull=True,
            status='ocr-complete',
        ).count()

        return Response({
            'used_bytes': used_bytes,
            'limit_bytes': 100 * 1024 * 1024 * 1024,
            'file_count': file_count,
            'by_type': by_type,
            'ocr_pages': ocr_pages,
        })


# ── Training data export ────────────────────────────���─────────────────────────

_LABEL_TO_CLASS = {
    'title': 0,
    'plain_text': 1,
    'figure': 2,
    'table': 3,
    'isolate_formula': 4,
}

_DATA_YAML = """\
path: .
train: images
val: images

names:
  0: title
  1: plain_text
  2: figure
  3: table
  4: isolate_formula
"""

_README = """\
# Layout Training Dataset

This dataset was exported from Content Extractor and contains human-corrected
layout annotations in YOLO format.

## Structure

    images/   — page images (JPEG, rendered at ~200 DPI)
    labels/   — YOLO annotation files (one per image)
    data.yaml — dataset configuration for ultralytics

Each label line: <class_id> <x_center> <y_center> <width> <height>
All values normalised to [0, 1].  Classes: 0=title 1=plain_text 2=figure 3=table 4=isolate_formula

## Fine-tuning on Google Colab (free GPU)

```python
!pip install doclayout_yolo
from doclayout_yolo import YOLOv10
model = YOLOv10("doclayout_yolo_docstructbench_imgsz1024.pt")
model.train(data="data.yaml", epochs=30, imgsz=1024, batch=4, lr0=0.0005)
# Find trained weights at: runs/detect/train/weights/best.pt
```

Replace the weights file on your server to use the improved model.
"""


class TrainingDataExportView(APIView):

    def get(self, request):
        from .models import DocumentPage

        doc_ids_param = request.query_params.get('doc_ids')
        qs = DocumentPage.objects.filter(document__deleted_at__isnull=True)
        if doc_ids_param:
            ids = [i.strip() for i in doc_ids_param.split(',') if i.strip()]
            qs = qs.filter(document_id__in=ids)

        training_dir = settings.TRAINING_IMAGES_DIR
        buf = io.BytesIO()
        added = 0

        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for page in qs.iterator():
                blocks = page.layout_blocks
                if not blocks:
                    continue
                img_path = training_dir / f"{page.session_id}.jpg"
                if not img_path.exists():
                    continue

                image_w = page.image_w or 0
                image_h = page.image_h or 0
                if image_w <= 0 or image_h <= 0:
                    try:
                        with Image.open(img_path) as im:
                            image_w, image_h = im.size
                    except Exception:
                        continue

                annotation_lines = []
                for b in blocks:
                    label = b.get('label', '')
                    class_id = _LABEL_TO_CLASS.get(label)
                    if class_id is None:
                        continue
                    bbox = b.get('bbox', [])
                    if len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = bbox
                    x_center = ((x1 + x2) / 2) / image_w
                    y_center = ((y1 + y2) / 2) / image_h
                    width    = (x2 - x1) / image_w
                    height   = (y2 - y1) / image_h
                    annotation_lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

                if not annotation_lines:
                    continue

                stem = f"{page.document_id}_p{page.page_number}"
                zf.write(str(img_path), f"images/{stem}.jpg")
                zf.writestr(f"labels/{stem}.txt", "\n".join(annotation_lines) + "\n")
                added += 1

            if added == 0:
                return Response({'error': 'no annotated pages available'}, status=404)

            zf.writestr('data.yaml', _DATA_YAML)
            zf.writestr('README.md', _README)

        buf.seek(0)
        response = HttpResponse(buf.read(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="layout_training_data.zip"'
        return response


class TatrTrainingDataView(APIView):
    """
    GET /api/tatr-training-data/

    Export a ZIP of table crop images + COCO annotations for all table blocks
    where table_structure.source == "edited" (user-corrected dividers).
    These are ground-truth samples for TATR fine-tuning.
    """

    def get(self, request):
        import json
        from .models import DocumentPage

        crops_dir = settings.TATR_CROPS_DIR
        buf = io.BytesIO()
        added = 0

        coco_images = []
        coco_annotations = []
        image_id = 0
        annotation_id = 0

        qs = DocumentPage.objects.filter(document__deleted_at__isnull=True)

        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for page in qs.iterator():
                blocks = page.layout_blocks
                if not blocks:
                    continue

                for block in blocks:
                    if block.get('label') != 'table':
                        continue
                    ts = block.get('table_structure')
                    if not ts or ts.get('source') != 'edited':
                        continue

                    block_id = block.get('id')
                    crop_filename = f"{page.session_id}_block_{block_id}.jpg"
                    crop_path = crops_dir / crop_filename
                    if not crop_path.exists():
                        continue

                    try:
                        with Image.open(crop_path) as im:
                            crop_w, crop_h = im.size
                    except Exception:
                        continue

                    image_id += 1
                    coco_images.append({
                        "id": image_id,
                        "file_name": crop_filename,
                        "width": crop_w,
                        "height": crop_h,
                    })

                    row_dividers = ts.get('row_dividers', [])
                    col_dividers = ts.get('col_dividers', [])
                    header_rows  = ts.get('header_rows', 0)

                    row_boundaries = [0] + [round(d * crop_h) for d in row_dividers] + [crop_h]
                    col_boundaries = [0] + [round(d * crop_w) for d in col_dividers] + [crop_w]

                    # Row annotations (header rows get category 1, body rows get 2)
                    for i in range(len(row_boundaries) - 1):
                        y1 = row_boundaries[i]
                        y2 = row_boundaries[i + 1]
                        annotation_id += 1
                        coco_annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1 if i < header_rows else 2,
                            "bbox": [0, y1, crop_w, y2 - y1],
                            "area": crop_w * (y2 - y1),
                            "iscrowd": 0,
                        })

                    # Column annotations
                    for j in range(len(col_boundaries) - 1):
                        x1 = col_boundaries[j]
                        x2 = col_boundaries[j + 1]
                        annotation_id += 1
                        coco_annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 3,
                            "bbox": [x1, 0, x2 - x1, crop_h],
                            "area": (x2 - x1) * crop_h,
                            "iscrowd": 0,
                        })

                    # Whole-table annotation covering the entire crop
                    annotation_id += 1
                    coco_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 4,
                        "bbox": [0, 0, crop_w, crop_h],
                        "area": crop_w * crop_h,
                        "iscrowd": 0,
                    })

                    zf.write(str(crop_path), f"images/{crop_filename}")
                    added += 1

            if added == 0:
                return Response({'error': 'no edited table structures available'}, status=404)

            coco = {
                "categories": [
                    {"id": 1, "name": "table column header"},
                    {"id": 2, "name": "table row"},
                    {"id": 3, "name": "table column"},
                    {"id": 4, "name": "table"},
                ],
                "images": coco_images,
                "annotations": coco_annotations,
            }
            zf.writestr("annotations.json", json.dumps(coco, indent=2))

        buf.seek(0)
        response = HttpResponse(buf.read(), content_type='application/zip')
        response['Content-Disposition'] = 'attachment; filename="tatr_training_data.zip"'
        return response


# ── Structured content ────────────────────────────────────────────────────────

class PageStructureView(APIView):

    def post(self, request, doc_id, page_number):
        from .models import Document, DocumentPage
        from .ml import structure_parser

        try:
            doc = Document.objects.get(id=doc_id)
        except Document.DoesNotExist:
            return Response({'error': 'document not found'}, status=404)

        try:
            page = DocumentPage.objects.get(document=doc, page_number=page_number)
        except DocumentPage.DoesNotExist:
            return Response({'error': 'page not found'}, status=404)

        # Idempotent: return existing structured content if already parsed
        if page.structured_content is not None:
            return Response({'structured_content': page.structured_content})

        if not page.ocr_blocks:
            return Response({'error': 'no ocr_blocks to structure'}, status=400)

        structured = structure_parser.parse(page.ocr_blocks)
        page.structured_content = structured
        page.structure_status = 'auto_parsed'
        page.save(update_fields=['structured_content', 'structure_status'])

        return Response({'structured_content': structured})
