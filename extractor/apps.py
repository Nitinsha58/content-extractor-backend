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

        self._cleanup_stale_artifacts()

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

    def _cleanup_stale_artifacts(self):
        """At startup, remove debug_cache files not referenced by any DocumentPage."""
        from pathlib import Path
        from django.conf import settings

        try:
            from .models import DocumentPage
            live = set(
                DocumentPage.objects.exclude(session_id__isnull=True)
                .exclude(session_id="")
                .values_list("session_id", flat=True)
            )
        except Exception:
            return  # DB not ready yet (e.g. first migration)

        freed = 0
        for path in Path(settings.DEBUG_CACHE_DIR).glob("*.jpg"):
            if path.stem not in live:
                try:
                    freed += path.stat().st_size
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

        if freed:
            print(f"🧹 Cleaned {freed // (1024*1024)} MB of stale debug cache files")
