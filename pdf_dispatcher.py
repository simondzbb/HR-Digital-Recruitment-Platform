"""Dispatch a resume path to the correct extractor based on its extension.

Public entry points:
    - ``parse_resume(path)``         → returns Markdown/plain text or None
    - ``extract_resume_data(path)``  → returns a ``JSONResume`` or None
    - ``UnsupportedFormatError``     → raised for unrecognized extensions

Extends the existing PDF-only flow without duplicating LLM section
extraction. ``extract_resume_data`` reuses
``PDFHandler._extract_all_sections_separately`` so the per-section LLM
calls live in exactly one place (``pdf.py``).
"""

import logging
import os
from typing import Optional

from pdf import PDFHandler
from docx_parser import extract_text_from_docx
from doc import extract_text_from_doc
from image import extract_text_from_image

logger = logging.getLogger(__name__)


class UnsupportedFormatError(ValueError):
    """Raised when the resume file extension is not one we support.

    Subclasses ``ValueError`` so callers that already handle
    ``ValueError`` automatically catch this too.
    """


def parse_resume(path: str) -> Optional[str]:
    """Extract Markdown/plain text from a resume file.

    Format is chosen by extension:
        .pdf                  → PyMuPDF (existing pipeline)
        .docx                 → python-docx
        .doc                  → LibreOffice headless → PyMuPDF
        .png/.jpg/.jpeg       → Tesseract OCR
        .tif/.tiff            → Tesseract OCR (first frame only)

    Args:
        path: Filesystem path to a resume file.

    Returns:
        Markdown/plain text suitable for the LLM section extractor,
        or ``None`` on failure (missing file, corrupted file, missing
        system dependency).

    Raises:
        FileNotFoundError: when ``path`` does not exist on disk.
        UnsupportedFormatError: when the extension is not in the known
            set. The caller (typically ``score.py``) maps this to a
            friendly CLI error.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext not in {".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise UnsupportedFormatError(
            f"Unsupported resume format '{ext}'. "
            f"Supported: .pdf, .docx, .doc, .png, .jpg, .jpeg, .tif, .tiff"
        )

    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume file not found: {path}")

    if ext == ".pdf":
        handler = PDFHandler()
        return handler.extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    if ext == ".doc":
        return extract_text_from_doc(path)
    # ext is one of {".png", ".jpg", ".jpeg", ".tif", ".tiff"} by the
    # whitelist check above.
    return extract_text_from_image(path)


def extract_resume_data(path: str):
    """Top-level: extract text then run LLM section extraction.

    This is the new entry point that replaces
    ``PDFHandler.extract_json_from_pdf``. It returns a fully populated
    ``JSONResume`` (or ``None``) by chaining ``parse_resume`` with the
    shared ``PDFHandler._extract_all_sections_separately`` private method.

    Args:
        path: Filesystem path to a resume file in any supported format.

    Returns:
        A ``JSONResume`` on success, or ``None`` on any failure
        (text extraction failed, section extraction failed, unsupported
        format — the latter is logged but NOT raised here; see
        ``score.py`` for the friendly CLI error path).
    """
    from models import JSONResume  # local import avoids cycle

    text = parse_resume(path)
    if not text:
        logger.error(f"❌ Failed to extract text from {path}")
        return None

    handler = PDFHandler()
    logger.debug(f"🔄 Extracting all sections from {path}...")
    return handler._extract_all_sections_separately(text)