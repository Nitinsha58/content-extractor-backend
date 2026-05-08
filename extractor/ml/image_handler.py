"""
Save PIL image crops to MEDIA_ROOT/figures/ and return ImageBlock instances.
"""

import uuid
from pathlib import Path

from PIL import Image
from django.conf import settings

from schema import ImageBlock


def _figures_dir() -> Path:
    d = settings.MEDIA_ROOT / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_crop(crop: Image.Image, alt: str = "figure") -> ImageBlock:
    fname = f"{uuid.uuid4().hex}.png"
    path = _figures_dir() / fname
    crop.save(str(path), format="PNG")
    return ImageBlock(
        url=f"/media/figures/{fname}",
        alt=alt,
        width=crop.width,
        height=crop.height,
    )
