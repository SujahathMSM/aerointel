"""Optional OCR for scanned PDF pages.

Needs two things that are not part of the default install: the `tesseract`
program and PyMuPDF (`pip install -r requirements-ocr.txt`). Everything here is
imported lazily, so the rest of the app runs fine without either.

Each blank page is rendered to an image by PyMuPDF and handed to tesseract
through a pipe. Nothing is written to disk.
"""

import importlib.util
import re
import shutil
import subprocess

from app.core.logging import get_logger

logger = get_logger(__name__)

_LANGUAGE = re.compile(r"^[A-Za-z_]{2,12}(\+[A-Za-z_]{2,12})*$")
PAGE_TIMEOUT_SECONDS = 120
RENDER_DPI = 200


class OcrUnavailable(RuntimeError):
    """OCR was requested but tesseract or PyMuPDF is not installed."""


def ocr_available() -> bool:
    return shutil.which("tesseract") is not None and (
        importlib.util.find_spec("pymupdf") is not None
    )


def ocr_pages(data: bytes, numbers: list[int], language: str = "eng") -> dict[int, str]:
    """OCR the given 1-based pages of a PDF; returns {page_number: text}.

    A page that fails to render or OCR is logged and left out of the result
    (the caller then treats it as blank) instead of failing the whole file.
    """
    if not _LANGUAGE.match(language):
        raise ValueError(f"invalid OCR language: {language!r}")
    if not ocr_available():
        raise OcrUnavailable(
            "OCR needs the tesseract program and PyMuPDF "
            "(pip install -r requirements-ocr.txt)."
        )

    import pymupdf  # imported here so it stays optional

    results: dict[int, str] = {}
    with pymupdf.open(stream=data, filetype="pdf") as document:
        for number in numbers:
            try:
                png = document[number - 1].get_pixmap(dpi=RENDER_DPI).tobytes("png")
                completed = subprocess.run(
                    ["tesseract", "stdin", "stdout", "-l", language],
                    input=png,
                    capture_output=True,
                    timeout=PAGE_TIMEOUT_SECONDS,
                    check=True,
                )
                results[number] = completed.stdout.decode("utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001 - one bad page must not sink the file
                logger.warning("ocr failed for page", page=number, error=str(exc))
    return results
