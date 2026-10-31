"""PIL image helpers with HEIC/HEIF support (iPhone photos)."""
from __future__ import annotations

from PIL import Image

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - pillow-heif is optional
    pass


def open_image(path) -> Image.Image:
    """Open an image as RGB, including HEIC/HEIF if pillow-heif is installed."""
    return Image.open(path).convert("RGB")
