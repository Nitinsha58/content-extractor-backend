"""
Django app config for extractor. Preloads ML models at startup.
"""

from django.apps import AppConfig


class ExtractorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'extractor'

    def ready(self):
        """Preload ML models when Django app is ready."""
        import sys
        from pathlib import Path

        # Add ml/ to path so we can import ML modules
        ml_dir = Path(__file__).parent / 'ml'
        if str(ml_dir) not in sys.path:
            sys.path.insert(0, str(ml_dir))

        try:
            from . import ml_state
            from .ml.layout import ExamLayoutParser
            from .ml.ocr_pipeline import init_models

            print("🔄 Initializing ExamLayoutParser...")
            ml_state.layout_parser = ExamLayoutParser()
            ml_state.layout_parser._load_model()
            print("✅ Layout model loaded")

            print("🔄 Initializing OCR models...")
            init_models()
            print("✅ OCR models loaded")

            print("🔄 Initializing TATR model...")
            from .ml.tatr import init_tatr
            init_tatr()
            print("✅ TATR model loaded")

        except Exception as e:
            print(f"⚠️  Warning: Failed to preload ML models: {e}")
            print("   Server will continue but extraction will fail.")
            import traceback
            traceback.print_exc()
