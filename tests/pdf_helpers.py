"""Build small real PDFs for tests, with no extra libraries.

`make_pdf` writes a minimal but valid PDF whose pages carry a text layer, so
pypdf reads it exactly as it would read a real manual. `make_scanned_pdf`
writes pages that are only a picture of text (no text layer), which is what a
scan looks like; it needs PyMuPDF and is skipped by the tests when absent.
"""


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """One page per entry; each entry is a list of text lines, top to bottom."""
    objects: list[bytes] = []

    def add(body: str) -> int:
        objects.append(body.encode("latin-1"))
        return len(objects)

    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids, kids = [], []
    for lines in pages:
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 780 Td"]
        for line in lines:
            ops.append(f"({_escape(line)}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops)
        content_id = add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        page_ids.append(content_id)
    pages_id = len(objects) + len(pages) + 1
    for content_id in page_ids:
        page_id = add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        kids.append(f"{page_id} 0 R")
    assert (
        add(f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>")
        == pages_id
    )
    catalog_id = add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def make_scanned_pdf(page_texts: list[str]) -> bytes:
    """Pages that are only a bitmap of the text: no text layer at all."""
    import pymupdf

    with pymupdf.open() as source, pymupdf.open() as scan:
        for text in page_texts:
            page = source.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 570, 750), text, fontsize=16)
        for page in source:
            image = page.get_pixmap(dpi=150).tobytes("png")
            blank = scan.new_page(width=page.rect.width, height=page.rect.height)
            blank.insert_image(blank.rect, stream=image)
        return scan.tobytes()
