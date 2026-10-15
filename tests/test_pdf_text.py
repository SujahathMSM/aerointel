"""PDF reading, cleaning and chunking: no database, no Ollama."""

import io
import itertools

import pytest
from pypdf import PdfWriter

from app.services.ocr import ocr_available, ocr_pages
from app.services.pdf_text import (
    PageText,
    PdfError,
    PdfNoTextError,
    chunk_pages,
    chunk_text,
    extract_pages,
    find_page_label,
    join_lines,
    split_sentences,
    strip_repeated_edges,
)
from tests.pdf_helpers import make_pdf, make_scanned_pdf

# ------------------------------------------------------------ page labels


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        (["9/8/98 AC 43.13-1B", "Par 3-2 Page 3-3", "Body text here."], "3-3"),
        (["Body text.", "More body.", "Page 112"], "112"),
        (["Body text.", "More body.", "Page xii"], "xii"),
        (["Body text.", "More body.", "7-45"], "7-45"),  # bare footer number
        (["Body text.", "More body.", "iv"], "iv"),
        (["Just prose, no number anywhere on this page."], None),
        ([], None),
    ],
)
def test_find_page_label(lines, expected):
    assert find_page_label(lines) == expected


def test_page_label_ignores_the_word_page_in_the_middle_of_the_body():
    body = [
        "intro line",
        "a",
        "b",
        "c",
        "d",
        "See Page 9-9 for details.",
        "e",
        "f",
        "g",
        "h",
    ]
    assert find_page_label(body) is None  # only the first/last lines are looked at


# ----------------------------------------------------- headers and footers


def _pages(
    n: int, header: str = "9/8/98 AC 43.13-1B", footer: str = "Par {i} Page {i}"
):
    return [
        [header, f"Body of page {i} talks about hydraulics.", footer.format(i=i)]
        for i in range(1, n + 1)
    ]


def test_repeated_header_and_footer_are_removed_even_though_digits_change():
    cleaned = strip_repeated_edges(_pages(10))
    assert all(len(lines) == 1 for lines in cleaned)
    assert cleaned[3] == ["Body of page 4 talks about hydraulics."]


def test_the_same_words_in_the_body_are_not_removed():
    pages = _pages(10)
    pages[4] = [
        pages[4][0],
        "a",
        "b",
        "c",
        "9/8/98 AC 43.13-1B",  # mid-page, not an edge line
        "d",
        "e",
        "f",
        pages[4][2],
    ]
    cleaned = strip_repeated_edges(pages)
    assert "9/8/98 AC 43.13-1B" in cleaned[4]


def test_a_line_on_only_a_few_pages_is_content_not_a_header():
    pages = _pages(20, header="Unique heading")
    pages[0][0] = "Chapter 1"
    pages[1][0] = "Chapter 2"
    cleaned = strip_repeated_edges(pages)
    assert cleaned[0][0] == "Chapter 1"
    assert cleaned[1][0] == "Chapter 2"


def test_short_documents_keep_their_edges():
    # Two pages cannot establish a "repeating" header.
    assert strip_repeated_edges(_pages(2)) == _pages(2)


# ------------------------------------------------------------- line joining


def test_join_lines_undoes_hyphenation_only_before_lowercase():
    assert join_lines(["The inspec-", "tion is due."]) == "The inspection is due."
    assert join_lines(["Per Part-", "145 rules."]) == "Per Part- 145 rules."


def test_join_lines_collapses_whitespace():
    assert join_lines(["  a   b ", "", "c\t d  "]) == "a b c d"


# ----------------------------------------------------------------- chunking


SENTENCES = [
    f"Sentence number {i} describes one inspection step in detail." for i in range(40)
]
TEXT = " ".join(SENTENCES)


def test_split_sentences():
    assert split_sentences("One. Two! Three? Four") == [
        "One.",
        "Two!",
        "Three?",
        "Four",
    ]


def test_chunks_respect_the_size_limit_and_end_on_sentences():
    chunks = chunk_text(TEXT, max_chars=300, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.endswith(".") for c in chunks)


def test_every_sentence_lands_in_some_chunk():
    chunks = chunk_text(TEXT, max_chars=300, overlap_chars=60)
    joined = " ".join(chunks)
    assert all(s in joined for s in SENTENCES)


def test_overlap_repeats_the_end_of_the_previous_chunk():
    chunks = chunk_text(TEXT, max_chars=300, overlap_chars=120)
    for before, after in itertools.pairwise(chunks):
        before_sentences = split_sentences(before)
        # the next chunk opens with sentences taken from the end of this one
        assert split_sentences(after)[0] in before_sentences
        assert before_sentences[-1] in split_sentences(after)


def test_no_overlap_means_no_repeated_sentences():
    chunks = chunk_text(TEXT, max_chars=300, overlap_chars=0)
    seen = [s for c in chunks for s in split_sentences(c)]
    assert len(seen) == len(set(seen))


def test_the_last_chunk_is_never_just_overlap():
    for size in range(150, 600, 37):
        chunks = chunk_text(TEXT, max_chars=size, overlap_chars=size // 2)
        last_sentences = split_sentences(chunks[-1])
        previous = split_sentences(chunks[-2]) if len(chunks) > 1 else []
        assert any(s not in previous for s in last_sentences)


def test_a_sentence_longer_than_the_limit_is_split_at_spaces():
    long = "word " * 200
    chunks = chunk_text(long.strip(), max_chars=100, overlap_chars=0)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == long.split()


def test_empty_text_and_bad_limit():
    assert chunk_text("", 100, 10) == []
    with pytest.raises(ValueError):
        chunk_text("text", 0, 0)


def test_chunking_is_deterministic():
    assert chunk_text(TEXT, 300, 60) == chunk_text(TEXT, 300, 60)


def test_chunks_never_cross_pages_and_keep_their_page_and_label():
    pages = [
        PageText(1, "1-1", "First page sentence about brakes. " * 12),
        PageText(2, "1-2", "Second page sentence about engines. " * 12),
    ]
    chunks = chunk_pages(pages, max_chars=200, overlap_chars=40)
    assert {c.page for c in chunks} == {1, 2}
    for c in chunks:
        assert ("brakes" in c.text) == (c.page == 1)
        assert ("engines" in c.text) == (c.page == 2)
        assert c.page_label == f"1-{c.page}"
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_nearly_empty_pages_make_no_chunks():
    pages = [PageText(1, None, "Fig 3."), PageText(2, None, "Real content " * 20)]
    chunks = chunk_pages(pages)
    assert {c.page for c in chunks} == {2}


# ------------------------------------------------------- reading real PDFs


def test_extract_pages_reads_text_labels_and_strips_headers():
    pdf = make_pdf(
        [
            [
                "9/8/98 AC 43.13-1B",
                f"Page 2-{i}",
                "First line of the body, which differs on every page.",
                f"Inspect the hydraulic filter on step {i}, then the pump.",
                "Third body line completes the paragraph for this page.",
                "Fourth body line keeps the page long enough to have a middle.",
            ]
            for i in range(1, 7)
        ]
    )
    pages = extract_pages(pdf)
    assert [p.number for p in pages] == [1, 2, 3, 4, 5, 6]
    assert [p.label for p in pages] == [f"2-{i}" for i in range(1, 7)]
    assert pages[2].text.startswith("First line of the body")
    assert "step 3" in pages[2].text
    assert "AC 43.13" not in pages[2].text
    assert "Page 2-3" not in pages[2].text
    assert not any(p.ocr for p in pages)


def test_garbage_is_a_clear_error_not_a_crash():
    with pytest.raises(PdfError):
        extract_pages(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(PdfError):
        extract_pages(b"hello")


def test_password_protected_pdf_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(PdfError, match="password"):
        extract_pages(buffer.getvalue())


def test_pdf_without_text_layer_asks_for_ocr():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(PdfNoTextError, match="OCR"):
        extract_pages(buffer.getvalue())


def test_ocr_fills_only_the_blank_pages_and_marks_them():
    text_pdf = make_pdf(
        [["Page 1-1", "This page has a real text layer with enough words to count."]]
    )
    calls = []

    def fake_ocr(data, numbers, language):
        calls.append((numbers, language))
        return {
            n: "Recognised text from the scan of this page, long enough."
            for n in numbers
        }

    # one text page: no blank pages, so OCR must not be called
    assert not extract_pages(text_pdf, ocr=fake_ocr)[0].ocr
    assert calls == []

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    pages = extract_pages(buffer.getvalue(), ocr=fake_ocr, ocr_language="eng+deu")
    assert calls == [([1], "eng+deu")]
    assert pages[0].ocr
    assert pages[0].text.startswith("Recognised text")


def test_ocr_that_finds_nothing_is_still_a_no_text_error():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    with pytest.raises(PdfNoTextError):
        extract_pages(buffer.getvalue(), ocr=lambda d, n, lang: {1: "ab"})


# ---------------------------------------------------------- real OCR engine


needs_ocr = pytest.mark.skipif(
    not ocr_available(), reason="needs the tesseract program and PyMuPDF"
)


@needs_ocr
def test_real_ocr_reads_a_scanned_page():
    scan = make_scanned_pdf(
        ["Inspect the hydraulic reservoir sight glass before first flight of the day."]
    )
    with pytest.raises(PdfNoTextError):
        extract_pages(scan)  # no text layer without OCR

    pages = extract_pages(scan, ocr=ocr_pages)
    assert pages[0].ocr
    assert "hydraulic reservoir" in pages[0].text.lower()


@needs_ocr
def test_ocr_rejects_a_language_that_could_inject_arguments():
    with pytest.raises(ValueError):
        ocr_pages(b"%PDF-", [1], "eng -c tessedit_write_images=1")
