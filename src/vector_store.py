import os
import json

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.embeddings import get_embedding_model


VECTORSTORE_DIR = "vectorstore"
MANIFEST_FILE = os.path.join(VECTORSTORE_DIR, "manifest.json")


def build_vectorstore(chunks):
    """
    Builds a FAISS vectorstore from extracted text chunks.
    """

    if not chunks:
        raise ValueError(
            "No text chunks were created from the uploaded PDF. "
            "Please check PDF text extraction."
        )

    documents = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        # Skip empty chunks
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": chunk.get("source", "Unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_id": chunk.get("chunk_id", 0),
                },
            )
        )

    if not documents:
        raise ValueError(
            "PDF was loaded, but no usable text was found after cleaning."
        )

    embedding_model = get_embedding_model()

    vectorstore = FAISS.from_documents(
        documents,
        embedding_model
    )

    return vectorstore


def save_vectorstore(
    vectorstore,
    file_names,
    path=VECTORSTORE_DIR
):
    """
    Saves FAISS vectorstore and the names of processed PDFs.
    """

    os.makedirs(path, exist_ok=True)

    vectorstore.save_local(path)

    manifest = {
        "files": sorted(file_names)
    }

    with open(
        MANIFEST_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2
        )


def load_vectorstore(path=VECTORSTORE_DIR):
    """
    Loads an existing FAISS vectorstore from disk.
    """

    if not vectorstore_exists(path):
        return None

    embedding_model = get_embedding_model()

    vectorstore = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    return vectorstore


def vectorstore_exists(path=VECTORSTORE_DIR):
    """
    Checks whether a saved FAISS index exists.
    """

    index_file = os.path.join(
        path,
        "index.faiss"
    )

    return os.path.exists(index_file)


def get_saved_manifest():
    """
    Returns the list of PDFs stored in the manifest.
    """

    if not os.path.exists(MANIFEST_FILE):
        return None

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    return manifest.get("files")


def manifest_matches(file_names):
    """
    Checks whether the uploaded PDFs match
    the PDFs used to create the saved vectorstore.
    """

    saved_files = get_saved_manifest()

    if saved_files is None:
        return False

    return sorted(file_names) == sorted(saved_files)