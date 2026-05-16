"""
URL routing for extractor app.
"""

from django.urls import path
from . import views

urlpatterns = [
    # ── ML processing ─────────────────────────────────────────────────────────
    path('api/debug/layout/', views.LayoutView.as_view(), name='layout'),
    path('api/debug/ocr/', views.OcrView.as_view(), name='ocr'),
    path('api/debug/ocr/stream/', views.OcrStreamView.as_view(), name='ocr_stream'),
    path('api/debug/crop/', views.CropView.as_view(), name='crop'),
    path('api/debug/table-structure/', views.TableStructureView.as_view(), name='table_structure'),
    path('api/debug/table-cell-types/', views.TableCellTypesView.as_view(), name='table_cell_types'),
    path('api/debug/session-image/<str:session_id>/', views.SessionImageView.as_view(), name='session_image'),
    path('api/export/', views.ExportView.as_view(), name='export'),

    # ── Document CRUD ─────────────────────────────────────────────────────────
    path('api/documents/', views.DocumentListCreateView.as_view(), name='document_list'),
    path('api/documents/<uuid:doc_id>/', views.DocumentDetailView.as_view(), name='document_detail'),
    path('api/documents/<uuid:doc_id>/pages/<int:page_number>/', views.PageSaveView.as_view(), name='page_save'),
    path('api/documents/<uuid:doc_id>/pages/<int:page_number>/structure/', views.PageStructureView.as_view(), name='page_structure'),
    path('api/figures/upload-s3/', views.FigureS3UploadView.as_view(), name='figure_upload_s3'),

    # ── Document actions ──────────────────────────────────────────────────────
    path('api/documents/<uuid:doc_id>/star/', views.DocumentStarView.as_view(), name='document_star'),
    path('api/documents/<uuid:doc_id>/move/', views.DocumentMoveView.as_view(), name='document_move'),
    path('api/documents/<uuid:doc_id>/tags/', views.DocumentTagsView.as_view(), name='document_tags'),
    path('api/documents/<uuid:doc_id>/restore/', views.DocumentRestoreView.as_view(), name='document_restore'),

    # ── Folders ───────────────────────────────────────────────────────────────
    path('api/folders/', views.FolderListCreateView.as_view(), name='folder_list'),
    path('api/folders/<uuid:folder_id>/', views.FolderDetailView.as_view(), name='folder_detail'),

    # ── Tags ──────────────────────────────────────────────────────────────────
    path('api/tags/', views.TagListCreateView.as_view(), name='tag_list'),
    path('api/tags/<uuid:tag_id>/', views.TagDetailView.as_view(), name='tag_detail'),

    # ── Dashboard data ────────────────────────────────────────────────────────
    path('api/activity/', views.ActivityListView.as_view(), name='activity'),
    path('api/storage/', views.StorageView.as_view(), name='storage'),

    # ── Training data export ──────────────────────────────────────────────────
    path('api/training-data/export/', views.TrainingDataExportView.as_view(), name='training_export'),
    path('api/tatr-training-data/', views.TatrTrainingDataView.as_view(), name='tatr_training_export'),
]
