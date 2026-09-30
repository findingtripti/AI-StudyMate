from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages_data, chunk_size=800, chunk_overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ],
        keep_separator=True
    )

    all_chunks = []
    chunk_counter = 0

    for page in pages_data:
        text = page.get("text", "").strip()

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for chunk_text in page_chunks:
            cleaned_text = " ".join(chunk_text.split()).strip()

            if len(cleaned_text) < 40:
                continue

            all_chunks.append({
                "text": cleaned_text,
                "source": page.get("source", "Unknown"),
                "page": page.get("page", 0),
                "chunk_id": chunk_counter
            })

            chunk_counter += 1

    return all_chunks