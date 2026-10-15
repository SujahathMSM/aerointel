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
