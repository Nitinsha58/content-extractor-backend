"""
WSGI config for ContentExtractorBackend project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ContentExtractorBackend.settings')

application = get_wsgi_application()
