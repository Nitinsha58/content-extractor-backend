"""
Management command: cleanup_artifacts

Removes disk artifacts that are no longer referenced by any DocumentPage
in the database. Safe to run at any time — only deletes files whose
session_id does not appear in the pages table.

Usage:
    python manage.py cleanup_artifacts
    python manage.py cleanup_artifacts --dry-run
    python manage.py cleanup_artifacts --older-than 7   # days
"""

import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from extractor.models import DocumentPage


class Command(BaseCommand):
    help = "Remove orphaned .debug_cache, training_images, and tatr_crops files"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would be deleted without deleting",
        )
        parser.add_argument(
            "--older-than",
            type=int,
            default=0,
            metavar="DAYS",
            help="Only delete files older than DAYS days (0 = all orphans)",
        )

    def handle(self, **options):
        dry_run = options["dry_run"]
        older_than_days = options["older_than"]
        cutoff = time.time() - older_than_days * 86400 if older_than_days > 0 else None

        live_sessions = set(
            DocumentPage.objects.exclude(session_id__isnull=True)
            .exclude(session_id="")
            .values_list("session_id", flat=True)
        )

        dirs = {
            "debug_cache":      (settings.DEBUG_CACHE_DIR,      "*.jpg"),
            "training_images":  (settings.TRAINING_IMAGES_DIR,  "*.jpg"),
            "tatr_crops":       (settings.TATR_CROPS_DIR,        "*.jpg"),
        }

        total_freed = 0
        total_deleted = 0

        for label, (directory, pattern) in dirs.items():
            freed = 0
            deleted = 0
            for path in Path(directory).glob(pattern):
                # Extract session_id from filename (stem for debug/training; prefix for tatr)
                stem = path.stem
                # tatr crops are named {session_id}_block_{block_id}
                session_id = stem.split("_block_")[0] if "_block_" in stem else stem

                if session_id in live_sessions:
                    continue

                if cutoff and path.stat().st_mtime > cutoff:
                    continue

                size = path.stat().st_size
                if dry_run:
                    self.stdout.write(f"  [dry-run] would delete {path.name} ({size // 1024}KB)")
                else:
                    path.unlink(missing_ok=True)
                freed += size
                deleted += 1

            label_mb = freed / (1024 * 1024)
            verb = "Would free" if dry_run else "Freed"
            self.stdout.write(
                f"{label}: {deleted} files, {verb} {label_mb:.1f} MB"
            )
            total_freed += freed
            total_deleted += deleted

        total_mb = total_freed / (1024 * 1024)
        verb = "Would free" if dry_run else "Freed"
        self.stdout.write(
            self.style.SUCCESS(
                f"\nTotal: {total_deleted} files, {verb} {total_mb:.1f} MB"
            )
        )
