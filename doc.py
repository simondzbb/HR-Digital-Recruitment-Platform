"""Extract text from legacy .doc (Microsoft Word 97-2003) resume files.

Bridges through headless LibreOffice (``soffice``) to convert .doc to PDF,
then reuses the PyMuPDF + ``pymupdf_rag.to_markdown`` pipeline already used
by the PDF path.

LibreOffice startup is slow on cold caches; we cap the conversion at
``SOFFICE_TIMEOUT_SECONDS`` so the CLI never hangs silently. All LibreOffice
errors (missing binary, timeout, conversion failure, missing output file)
map to ``None`` with a clear log message — same contract as
``pdf.extract_text_from_pdf``.

Requires LibreOffice 7+ installed locally and ``soffice`` (or
``soffice.exe`` on Windows) on ``PATH``.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import pymupdf
from pymupdf_rag import to_markdown

logger = logging.getLogger(__name__)

# Hard timeout for the soffice invocation. LibreOffice cold-startup is
# ~5–10s; conversion of a one-page resume typically fits well under 30s.
# 60s is generous and prevents indefinite CLI hangs.
SOFFICE_TIMEOUT_SECONDS = 60


def _soffice_path() -> str:
    """Resolve the soffice executable via shutil.which.

    On Windows the binary is named ``soffice.exe``; on macOS/Linux it is
    ``soffice``. Falls back to the bare name so subprocess can try PATH
    itself and produce a FileNotFoundError we can catch.
    """
    return shutil.which("soffice") or shutil.which("soffice.exe") or "soffice"


def extract_text_from_doc(path: str) -> Optional[str]:
    """Convert a .doc file to PDF via LibreOffice and extract Markdown text.

    Steps:
        1. Validate the file exists and is non-empty.
        2. Create a temporary directory for soffice's output.
        3. Invoke ``soffice --headless --convert-to pdf --outdir <tmp> <path>``.
        4. Open the resulting PDF with PyMuPDF and call ``to_markdown()``.
        5. Always remove the temporary directory.

    Args:
        path: Filesystem path to a .doc file.

    Returns:
        Markdown text from the converted PDF, or ``None`` on any failure
        (missing file, empty file, soffice not on PATH, conversion timeout,
        conversion error, missing output PDF).
    """
    if not os.path.exists(path):
        logger.error(f"DOC file not found: {path}")
        return None

    if os.path.getsize(path) == 0:
        logger.error(f"DOC file is empty: {path}")
        return None

    soffice = _soffice_path()
    tmpdir = tempfile.mkdtemp(prefix="hiring_agent_doc_")
    try:
        try:
            subprocess.run(
                [
                    soffice,
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    tmpdir,
                    path,
                ],
                check=True,
                timeout=SOFFICE_TIMEOUT_SECONDS,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.error(
                "LibreOffice (soffice) not found on PATH. "
                "Install LibreOffice to enable .doc support. "
                "See README for platform instructions."
            )
            return None
        except subprocess.TimeoutExpired:
            logger.error(
                f"LibreOffice conversion timed out after "
                f"{SOFFICE_TIMEOUT_SECONDS}s for {path}"
            )
            return None
        except subprocess.CalledProcessError as e:
            stderr_snippet = (
                e.stderr.decode(errors="ignore")[:500] if e.stderr else ""
            )
            logger.error(
                f"LibreOffice failed to convert {path} to PDF: "
                f"exit={e.returncode}, stderr={stderr_snippet}"
            )
            return None

        base = os.path.splitext(os.path.basename(path))[0]
        converted_pdf = os.path.join(tmpdir, base + ".pdf")
        if not os.path.exists(converted_pdf):
            logger.error(
                f"LibreOffice did not produce expected PDF at {converted_pdf}"
            )
            return None

        with pymupdf.open(converted_pdf) as pdf_doc:
            text = to_markdown(pdf_doc, pages=range(pdf_doc.page_count))
            logger.debug(
                f"Extracted text from DOC: {len(text) if text else 0} characters"
            )
            return text
    except Exception as e:
        logger.error(f"An error occurred while reading the DOC: {e}")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)