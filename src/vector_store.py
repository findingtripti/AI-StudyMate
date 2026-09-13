import os
import json
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from src.embeddings import get_embedding_model

VECTORSTORE_DIR = "vectorstore"
MANIFEST_FILE = os.path.join(VECTORSTORE_DIR, "manifest.json")


def build_vectorstore(chunks):
    """Builds a new FAISS vectorstore from a list of chunk dicts."""
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }
        )
        for chunk in chunks
    ]

    embedding_model = get_embedding_model()
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore


def save_vectorstore(vectorstore, file_names, path=VECTORSTORE_DIR):
    """
    Saves the FAISS vectorstore to disk, along with a manifest
    listing which files were used to build it.
    """
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)

    manifest = {"files": sorted(file_names)}
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f)


def load_vectorstore(path=VECTORSTORE_DIR):
    """Loads a previously saved FAISS vectorstore from disk."""
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
    """Checks whether a saved vectorstore already exists on disk."""
    index_file = os.path.join(path, "index.faiss")
    return os.path.exists(index_file)


def get_saved_manifest():
    """Returns the list of file names used in the saved vectorstore, or None."""
    if not os.path.exists(MANIFEST_FILE):
        return None

    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    return manifest.get("files")


def manifest_matches(file_names):
    """Checks if the given file names match the saved manifest exactly."""
    saved_files = get_saved_manifest()
    if saved_files is None:
        return False
    return sorted(file_names) == saved_files