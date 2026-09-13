import streamlit as st
from src.pdf_loader import load_multiple_pdfs
from src.text_splitter import chunk_pages
from src.vector_store import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    manifest_matches,
)
from src.rag_pipeline import generate_answer

# Page configuration
st.set_page_config(
    page_title="AI StudyMate",
    page_icon="📚",
    layout="wide"
)

# ---- Initialize session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = None


# ---- Sidebar ----
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

    if uploaded_files:
        current_file_names = [f.name for f in uploaded_files]

        # Case 1: Already processed in THIS session — do nothing (fastest)
        if tuple(current_file_names) == st.session_state.processed_files:
            st.info(f"✅ {len(uploaded_files)} file(s) already processed")

        # Case 2: Not processed this session, but a matching vectorstore exists on disk
        elif manifest_matches(current_file_names):
            with st.spinner("Loading saved vectorstore from disk..."):
                st.session_state.vectorstore = load_vectorstore()
                st.session_state.processed_files = tuple(current_file_names)
            st.success(f"✅ Loaded existing vectorstore for {len(uploaded_files)} file(s) from disk")

        # Case 3: New/different files — process from scratch
        else:
            with st.spinner("Reading PDFs..."):
                pages_data = load_multiple_pdfs(uploaded_files)

            with st.spinner("Splitting text into chunks..."):
                chunks = chunk_pages(pages_data)

            with st.spinner("Generating embeddings and building vectorstore... (this may take a minute)"):
                vectorstore = build_vectorstore(chunks)
                save_vectorstore(vectorstore, current_file_names)

            st.session_state.vectorstore = vectorstore
            st.session_state.processed_files = tuple(current_file_names)

            st.success(f"✅ {len(uploaded_files)} file(s) processed — {len(chunks)} chunks indexed")

        for file in uploaded_files:
            st.write(f"- {file.name}")
    else:
        st.info("No documents uploaded yet")
        st.session_state.vectorstore = None
        st.session_state.processed_files = None


# ---- Main Area ----
st.title("💬 Chat with your Documents")
st.caption("Upload PDFs from the sidebar, then ask questions about them.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            response_text = "⚠️ Please upload at least one PDF before asking questions."
            st.write(response_text)
        else:
            with st.spinner("Searching your documents..."):
                results = st.session_state.vectorstore.similarity_search(user_question, k=3)

            if not results:
                response_text = "I couldn't find enough information about this in the uploaded documents."
                st.write(response_text)
            else:
                with st.spinner("Generating answer..."):
                    try:
                        answer = generate_answer(user_question, results)
                    except ValueError as e:
                        answer = f"⚠️ {str(e)}"

                st.write(answer)

                with st.expander(f"📚 View Sources ({len(results)})"):
                    seen_pages = set()

                    for res in results:
                        source = res.metadata.get("source")
                        page = res.metadata.get("page")
                        page_key = (source, page)

                        # Skip exact duplicate (same file + same page) to avoid clutter
                        if page_key in seen_pages:
                            continue
                        seen_pages.add(page_key)

                        text_snippet = res.page_content.strip()[:220]

                        st.markdown(
                            f"""
                            <div style="
                                border: 1px solid rgba(250, 250, 250, 0.2);
                                border-radius: 8px;
                                padding: 12px 16px;
                                margin-bottom: 10px;
                                background-color: rgba(250, 250, 250, 0.03);
                            ">
                                <div style="font-weight: 600; margin-bottom: 6px;">
                                    📄 {source} &nbsp;·&nbsp; 📌 Page {page}
                                </div>
                                <div style="font-size: 0.85em; opacity: 0.8;">
                                    {text_snippet}...
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                response_text = answer

    st.session_state.messages.append({"role": "assistant", "content": response_text})