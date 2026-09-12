from langchain_huggingface import HuggingFaceEmbeddings

# Model ko ek baar load karke reuse karenge 
_embedding_model = None


def get_embedding_model():
    """
    Loads (or returns already-loaded) the HuggingFace embedding model.
    Using a module-level cache so the model loads only once per app run.
    """
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embedding_model


def embed_texts(texts):
    """
    Converts a list of text strings into a list of embedding vectors.

    Args:
        texts: List of strings (e.g. chunk texts)

    Returns:
        List of vectors (each vector is a list of floats)
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_query(query_text):
    """
    Converts a single query string (like a user's question) into an embedding vector.

    Args:
        query_text: A single string

    Returns:
        A single vector (list of floats)
    """
    model = get_embedding_model()
    return model.embed_query(query_text)