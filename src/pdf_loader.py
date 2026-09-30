import io
import os
import re
from collections import Counter

import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from pypdf import PdfReader


# ---------------------------------------------------------
# Tesseract configuration
# ---------------------------------------------------------

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# OCR settings
OCR_DPI = 200
OCR_LANGUAGE = "eng"


# ---------------------------------------------------------
# Text quality detection
# ---------------------------------------------------------

def _is_garbage_text(text):
    """
    Detects whether extracted PDF text is empty,
    extremely short, or mostly meaningless/repeated text.
    """

    if not text:
        return True

    text = text.strip()

    # Too little text
    if len(text) < 40:
        return True

    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())

    if len(words) < 8:
        return True

    unique_words = set(words)

    # Very low vocabulary usually indicates broken extraction
    if len(unique_words) <= 3 and len(words) >= 8:
        return True

    word_counts = Counter(words)
    most_common_word, most_common_count = word_counts.most_common(1)[0]

    # One word dominating the page is usually extraction noise
    if most_common_count / len(words) > 0.65:
        return True

    return False


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def _clean_line(line):
    """Cleans obvious extraction/OCR noise from one line."""

    line = line.strip()

    if not line:
        return ""

    # Remove excessive whitespace
    line = re.sub(r"\s+", " ", line)

    # Remove obvious repeated garbage tokens
    if len(line) <= 40:
        words = line.lower().split()

        if len(words) >= 4 and len(set(words)) == 1:
            return ""

    return line


def _clean_text(text):
    """
    Cleans extracted/OCR text while preserving
    meaningful study content.
    """

    if not text:
        return ""

    cleaned_lines = []

    for line in text.splitlines():
        line = _clean_line(line)

        if line:
            cleaned_lines.append(line)

    # Rebuild paragraphs
    text = "\n".join(cleaned_lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove excessive spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Repeated watermark/noise removal
# ---------------------------------------------------------

def _remove_repeated_lines_across_pages(page_texts):
    """
    Removes short lines that appear on almost every page.
    This is mainly useful for repeated watermarks, branding,
    URLs, or extraction noise.

    Important:
    We keep this conservative so normal headings are not
    accidentally removed.
    """

    if len(page_texts) < 3:
        return page_texts

    line_page_count = Counter()

    for text in page_texts:
        seen_on_page = set()

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            normalized = re.sub(r"\s+", " ", line.lower())

            # Only consider relatively short repeated lines
            if len(normalized) <= 80:
                seen_on_page.add(normalized)

        for line in seen_on_page:
            line_page_count[line] += 1

    threshold = max(3, int(len(page_texts) * 0.75))

    repeated_noise = {
        line
        for line, count in line_page_count.items()
        if count >= threshold
        and (
            len(line.split()) <= 6
            or "http" in line
            or "www." in line
            or "watermark" in line
        )
    }

    cleaned_pages = []

    for text in page_texts:
        lines = []

        for line in text.splitlines():
            normalized = re.sub(r"\s+", " ", line.strip().lower())

            if normalized in repeated_noise:
                continue

            lines.append(line)

        cleaned_pages.append("\n".join(lines).strip())

    return cleaned_pages


# ---------------------------------------------------------
# PyPDF extraction
# ---------------------------------------------------------

def _extract_with_pypdf(pdf_file):
    """
    Extracts text using pypdf.
    """

    try:
        pdf_file.seek(0)

        reader = PdfReader(pdf_file)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            pages.append(text)

        return pages

    except Exception:
        return []


# ---------------------------------------------------------
# PyMuPDF native extraction
# ---------------------------------------------------------

def _extract_with_pymupdf(pdf_file):
    """
    Uses PyMuPDF as a second extraction method.
    """

    try:
        pdf_file.seek(0)

        pdf_bytes = pdf_file.read()

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        pages = []

        for page in document:
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""

            pages.append(text)

        document.close()

        return pages

    except Exception:
        return []


# ---------------------------------------------------------
# OCR image preprocessing
# ---------------------------------------------------------

def _preprocess_image_for_ocr(image):
    """
    Improves PDF page image before OCR.
    """

    image = image.convert("L")

    # Improve contrast
    image = ImageOps.autocontrast(image)

    # Slight sharpening
    image = image.filter(ImageFilter.SHARPEN)

    return image


# ---------------------------------------------------------
# OCR one page
# ---------------------------------------------------------

def _ocr_page(pdf_document, page_number):
    """
    Converts one PDF page to an image and runs Tesseract OCR.
    """

    page = pdf_document[page_number]

    matrix = fitz.Matrix(
        OCR_DPI / 72,
        OCR_DPI / 72
    )

    pixmap = page.get_pixmap(
        matrix=matrix,
        alpha=False
    )

    image_bytes = pixmap.tobytes("png")

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    image = _preprocess_image_for_ocr(image)

    text = pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
        config="--psm 6"
    )

    return text


# ---------------------------------------------------------
# OCR entire PDF
# ---------------------------------------------------------

def _extract_with_ocr(pdf_file, bad_page_indexes=None):
    """
    OCR fallback for pages where normal extraction failed.
    """

    try:
        pdf_file.seek(0)

        pdf_bytes = pdf_file.read()

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

        pages = []

        for page_number in range(len(document)):

            if bad_page_indexes is not None:
                if page_number not in bad_page_indexes:
                    pages.append("")
                    continue

            try:
                text = _ocr_page(
                    document,
                    page_number
                )
            except Exception:
                text = ""

            pages.append(text)

        document.close()

        return pages

    except Exception:
        return []


# ---------------------------------------------------------
# Smart extraction
# ---------------------------------------------------------

def _extract_pages_smart(pdf_file):
    """
    Smart PDF extraction pipeline:

        1. pypdf
        2. PyMuPDF native extraction
        3. OCR only for bad pages
    """

    # -----------------------------
    # First attempt: pypdf
    # -----------------------------

    pypdf_pages = _extract_with_pypdf(pdf_file)

    # -----------------------------
    # Second attempt: PyMuPDF
    # -----------------------------

    pymupdf_pages = _extract_with_pymupdf(pdf_file)

    page_count = max(
        len(pypdf_pages),
        len(pymupdf_pages)
    )

    combined_pages = []
    bad_page_indexes = []

    for i in range(page_count):

        pypdf_text = (
            pypdf_pages[i]
            if i < len(pypdf_pages)
            else ""
        )

        pymupdf_text = (
            pymupdf_pages[i]
            if i < len(pymupdf_pages)
            else ""
        )

        # Prefer pypdf if it looks valid
        if not _is_garbage_text(pypdf_text):
            selected_text = pypdf_text

        # Otherwise use PyMuPDF
        elif not _is_garbage_text(pymupdf_text):
            selected_text = pymupdf_text

        # Otherwise OCR this page
        else:
            selected_text = ""
            bad_page_indexes.append(i)

        combined_pages.append(selected_text)

    # -----------------------------
    # OCR only bad pages
    # -----------------------------

    if bad_page_indexes:

        ocr_pages = _extract_with_ocr(
            pdf_file,
            bad_page_indexes=bad_page_indexes
        )

        for i in bad_page_indexes:

            if i < len(ocr_pages):
                ocr_text = ocr_pages[i]

                if not _is_garbage_text(ocr_text):
                    combined_pages[i] = ocr_text

    # -----------------------------
    # Clean extracted text
    # -----------------------------

    combined_pages = [
        _clean_text(text)
        for text in combined_pages
    ]

    # Remove repeated watermark/noise
    combined_pages = _remove_repeated_lines_across_pages(
        combined_pages
    )

    return combined_pages


# ---------------------------------------------------------
# Public function: load one PDF
# ---------------------------------------------------------

def load_pdf(uploaded_file):
    """
    Loads one uploaded PDF and returns page-level data.

    Output format:

    [
        {
            "text": "...",
            "source": "example.pdf",
            "page": 1
        },
        ...
    ]
    """

    source_name = uploaded_file.name

    page_texts = _extract_pages_smart(
        uploaded_file
    )

    pages_data = []

    for page_number, text in enumerate(
        page_texts,
        start=1
    ):

        text = text.strip()

        # Skip completely empty pages
        if not text:
            continue

        pages_data.append(
            {
                "text": text,
                "source": source_name,
                "page": page_number
            }
        )

    return pages_data


# ---------------------------------------------------------
# Public function: load multiple PDFs
# ---------------------------------------------------------

def load_multiple_pdfs(uploaded_files):
    """
    Loads multiple PDFs using the same smart pipeline.
    """

    all_pages = []

    for uploaded_file in uploaded_files:

        pages = load_pdf(
            uploaded_file
        )

        all_pages.extend(pages)

    return all_pages