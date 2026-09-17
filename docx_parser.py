"""Extract plain text from .docx (Office Open XML) resume files.

Uses python-docx to walk paragraphs and tables in document order, returning
text with paragraphs separated by blank lines so downstream Markdown consumers
see "paragraphs" the same way they do in the PDF path.

Mirrors the error-handling contract of ``pdf.extract_text_from_pdf``:
log on failure, return ``None`` on error, never raise to the caller.

This module is named ``docx_parser`` (not ``docx``) so it does not shadow
the third-party ``python-docx`` package when we do ``from docx import Document``.
"""

import logging
import os
from typing import Optional

import docx  # python-docx package — do NOT name this module "docx" or it shadows the import
from docx.document import Document as _Document

logger = logging.getLogger(__name__)


def extract_text_from_docx(path: str) -> Optional[str]:
    """Extract text from a .docx file.

    Iterates ``document.paragraphs`` first (preserves reading flow for the
    common case) and then ``document.tables`` (resume tables such as skills
    grids or project matrices). Table cells in a row are joined with `` | ``.

    Args:
        path: Filesystem path to a readable .docx file.

    Returns:
        Plain text with paragraphs separated by ``\\n\\n``, or ``None`` on
        any failure (missing file, corrupted zip, password-protected doc,
        permission error). The error is logged via ``logger.error``.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"DOCX file not found: {path}")

        document = _Document(path)

        parts: list[str] = []

        for para in document.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))

        result = "\n\n".join(parts)
        logger.debug(
            f"Extracted text from DOCX: {len(result) if result else 0} characters"
        )
        return result
    except Exception as e:
        logger.error(f"An error occurred while reading the DOCX: {e}")
        return None