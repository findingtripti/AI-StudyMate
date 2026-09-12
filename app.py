from src.embeddings import embed_texts
from src.text_splitter import chunk_pages
import streamlit as st
from src.pdf_loader import load_multiple_pdfs

# Page configuration 
st.set_page_config(
    page_title="AI StudyMate",
    page_icon="📚",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📚 AI StudyMate")
    st.markdown("Your Intelligent Study Assistant")
    st.divider()

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your study PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )
#temp test code
if uploaded_files:
     st.success(f"{len(uploaded_files)} file(s) uploaded")
     for file in uploaded_files:
            st.write(f"- {file.name}")

     
             # ---- TEMPORARY TEST CODE (Phase 3 + 4) ----
     st.divider()
     st.subheader("🧪 PDF Extraction Test")
     pages_data = load_multiple_pdfs(uploaded_files)
     st.write(f"Total pages extracted: {len(pages_data)}")

     if pages_data:
            st.write("Preview of first page extracted:")
            st.json({
                "source": pages_data[0]["source"],
                "page": pages_data[0]["page"],
                "text_preview": pages_data[0]["text"][:300]
            })
     st.divider()
     st.subheader("🧪 Chunking Test")
     chunks = chunk_pages(pages_data)
     st.write(f"Total chunks created: {len(chunks)}")

     if chunks:
         avg_chunk_len = sum(len(c["text"]) for c in chunks) / len(chunks)
         st.write(f"Average chunk length: {avg_chunk_len:.0f} characters")

         st.write("Preview of first 2 chunks:")
         for chunk in chunks[:2]:
             st.json({
                 "chunk_id": chunk["chunk_id"],
                 "source": chunk["source"],
                 "page": chunk["page"],
                 "text": chunk["text"]
             })
     st.divider()
     st.subheader("🧪 Embedding Test")
     with st.spinner("Loading embedding model and generating embeddings... (first time may take a minute)"):
         sample_chunks = [chunk["text"] for chunk in chunks[:3]]
         sample_embeddings = embed_texts(sample_chunks)

     st.write(f"Number of embeddings generated: {len(sample_embeddings)}")
     st.write(f"Embedding vector dimension: {len(sample_embeddings[0])}")
     st.write("Preview of first embedding vector (first 10 numbers):")
     st.code(sample_embeddings[0][:10])      


else:
    st.info("No documents uploaded yet")

# Main Area
st.title("💬 Chat with your Documents")
st.caption("Upload PDFs from the sidebar, then ask questions about them.")

# Initialize chat history in session_state (persists across reruns)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    # Placeholder response (real RAG logic will come in later phases)
    placeholder_response = "🔧 RAG pipeline is not connected yet. This is just the UI skeleton (Phase 2)."
    st.session_state.messages.append({"role": "assistant", "content": placeholder_response})
    with st.chat_message("assistant"):
        st.write(placeholder_response)

