"""Extract text from image resume files via local Tesseract OCR.

Wraps ``pytesseract.image_to_string`` over a Pillow-loaded image. Applies
minimal cleanup to the OCR output (strip trailing whitespace, drop form
feeds, collapse runs of blank lines) — the LLM section extractor handles
structure inference, so aggressive cleaning would only hide signal.

Mirrors the error-handling contract of ``pdf.extract_text_from_pdf``:
log on failure, return ``None`` on error, never raise to the caller.

Requires the Tesseract binary (``tesseract``) and the ``eng`` language pack
to be installed system-wide and on ``PATH``. See README for platform notes.
"""

import logging
import os
from typing import Optional

import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _clean_ocr_text(raw: str) -> str:
    """Lightly clean OCR output for downstream Markdown consumers.

    - Strip trailing whitespace per line.
    - Drop form-feed / page-break characters.
    - Collapse runs of empty lines into a single blank line.
    - Trim leading and trailing whitespace from the whole string.
    """
    if not raw:
        return ""
    lines = [line.rstrip() for line in raw.splitlines()]
    lines = [line.replace("\f", "") for line in lines]
    collapsed: list[str] = []
    blank = False
    for line in lines:
        if line == "":
            if not blank:
                collapsed.append("")
            blank = True
        else:
            collapsed.append(line)
            blank = False
    return "\n".join(collapsed).strip()


def extract_text_from_image(path: str) -> Optional[str]:
    """Extract text from an image via Tesseract OCR.

    Args:
        path: Filesystem path to a PNG/JPEG/TIFF file.

    Returns:
        Plain text recognized by Tesseract (no Markdown wrapping), or
        ``None`` on any failure (missing file, unsupported extension,
        Tesseract not installed, Pillow cannot decode the image, OCR
        pipeline error).

    Behavior notes:
        - Phone-camera resume photos often carry EXIF rotation tags;
          we transpose via ``ImageOps.exif_transpose`` so Tesseract sees
          the image in its intended orientation.
        - We convert to RGB to avoid a class of Tesseract bugs with
          palette/alpha PNGs.
        - Multi-page TIFFs are not iterated — Tesseract via pytesseract
          handles one frame at a time. We OCR the first frame and log a
          warning so the user knows additional pages were skipped.
    """
    if not os.path.exists(path):
        logger.error(f"Image file not found: {path}")
        return None

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTS:
        logger.error(f"Unsupported image extension: {ext}")
        return None

    try:
        with Image.open(path) as img:
            if getattr(img, "is_animated", False) or (
                hasattr(img, "n_frames") and img.n_frames > 1
            ):
                logger.warning(
                    f"Multi-page image detected ({path}); only the first "
                    "frame will be OCR'd."
                )

            img = ImageOps.exif_transpose(img).convert("RGB")

            try:
                raw = pytesseract.image_to_string(img, lang="eng")
            except pytesseract.TesseractNotFoundError:
                logger.error(
                    "Tesseract binary not found on PATH. "
                    "Install Tesseract OCR (e.g. `apt install tesseract-ocr`, "
                    "`brew install tesseract`) and ensure the `eng` language "
                    "pack is present. See README."
                )
                return None

        cleaned = _clean_ocr_text(raw)
        logger.debug(
            f"Extracted text from image: {len(cleaned) if cleaned else 0} characters"
        )
        return cleaned
    except UnidentifiedImageError as e:
        logger.error(f"Pillow cannot decode image {path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An error occurred while OCR'ing {path}: {e}")
        return None