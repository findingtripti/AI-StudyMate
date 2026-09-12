from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages_data, chunk_size=1000, chunk_overlap=200):
    """
    Splits page-level text into smaller overlapping chunks,
    while preserving source (filename) and page number metadata.

    Args:
        pages_data: List of dicts like {"text": ..., "source": ..., "page": ...}
        chunk_size: Max characters per chunk
        chunk_overlap: Characters shared between consecutive chunks

    Returns:
        A list of dicts like:
        [
            {
                "text": "...chunk content...",
                "source": "DBMS.pdf",
                "page": 3,
                "chunk_id": 0
            },
            ...
        ]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # tries paragraph/sentence breaks first
    )

    all_chunks = []
    chunk_counter = 0

    for page in pages_data:
        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            all_chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk_id": chunk_counter
            })
            chunk_counter += 1

    return all_chunks