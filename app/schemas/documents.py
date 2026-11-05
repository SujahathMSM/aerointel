from datetime import date

from pydantic import BaseModel


class DocumentResponse(BaseModel):
    document_id: int
    title: str
    filename: str
    pages: int
    chunks: int
    # Pages whose text came from OCR instead of the PDF's own text layer.
    ocr_pages: int
    # True if these exact bytes were already ingested; nothing new was stored.
    duplicate: bool
    # Which issue of the document this is ("AC 43.13-1B CHG 1"), the latest
    # date printed in it, and what stays the same across its revisions.
    revision: str | None = None
    revision_date: date | None = None
    document_key: str | None = None
    superseded: bool = False
