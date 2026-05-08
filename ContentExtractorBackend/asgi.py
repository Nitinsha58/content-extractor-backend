"""
ASGI config for ContentExtractorBackend project.
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ContentExtractorBackend.settings')

application = get_asgi_application()
