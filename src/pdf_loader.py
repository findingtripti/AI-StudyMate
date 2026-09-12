from pypdf import PdfReader


def load_pdf(uploaded_file):
    """
    Reads a single PDF file and extracts text page-by-page.

    Args:
        uploaded_file: A file-like object (from Streamlit's file_uploader)

    Returns:
        A list of dictionaries, one per page, like:
        [
            {
                "text": "...page content...",
                "source": "DBMS.pdf",
                "page": 1
            },
            ...
        ]
    """
    reader = PdfReader(uploaded_file)
    filename = uploaded_file.name

    pages_data = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        # Skip pages that have no extractable text (e.g. blank or scanned pages)
        if text and text.strip():
            pages_data.append({
                "text": text,
                "source": filename,
                "page": page_number
            })

    return pages_data


def load_multiple_pdfs(uploaded_files):
    """
    Loads multiple PDFs and combines their page data into one list.

    Args:
        uploaded_files: A list of file-like objects

    Returns:
        A combined list of page dictionaries from all PDFs
    """
    all_pages = []

    for uploaded_file in uploaded_files:
        pages = load_pdf(uploaded_file)
        all_pages.extend(pages)

    return all_pages