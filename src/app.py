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
from src.summarizer import summarize_document, generate_suggested_questions
from src.unit_detector import get_available_units, get_unit_text
from src.quiz_generator import generate_quiz

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

if "document_summary" not in st.session_state:
    st.session_state.document_summary = None

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "pages_data" not in st.session_state:
    st.session_state.pages_data = None

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_title" not in st.session_state:
    st.session_state.quiz_title = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


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
            

            st.session_state.pages_data = pages_data

            with st.spinner("Splitting text into chunks..."):
              chunks = chunk_pages(pages_data)

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

        st.divider()

        if st.button("📝 Summarize Document(s)"):
            with st.spinner("Reading documents for summarization..."):
                pages_data = load_multiple_pdfs(uploaded_files)
                chunks = chunk_pages(pages_data)

            progress_bar = st.progress(0, text="Summarizing sections...")

            def update_progress(current, total):
                progress_bar.progress(current / total, text=f"Summarizing section {current} of {total}...")

            with st.spinner("Combining into final summary..."):
                summary = summarize_document(chunks, progress_callback=update_progress)

            progress_bar.empty()
            st.session_state.document_summary = summary

        if st.button("💡 Suggest Questions"):
            with st.spinner("Generating suggested questions..."):
                # Use a handful of sample chunks instead of the full summary (much faster)
                sample_pages = load_multiple_pdfs(uploaded_files)
                sample_chunks = chunk_pages(sample_pages)
                
                total = len(sample_chunks)
                if total <= 8:
                    picked = sample_chunks
                else:
                    step = total // 8
                    picked = [sample_chunks[i] for i in range(0, total, step)][:8]
                sample_text = "\n\n".join(c["text"] for c in picked)

                questions = generate_suggested_questions(sample_text)
                st.session_state.suggested_questions = questions

    else:
        st.info("No documents uploaded yet")
        st.session_state.vectorstore = None
        st.session_state.processed_files = None


# ---- Main Area ----
st.title("💬 Chat with your Documents")
st.caption("Upload PDFs from the sidebar, then ask questions about them.")

if st.session_state.document_summary:
    with st.expander("📝 Document Summary", expanded=True):
        st.markdown(st.session_state.document_summary)
    st.divider()

if st.session_state.suggested_questions:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(2)
    for i, question in enumerate(st.session_state.suggested_questions):
        col = cols[i % 2]
        with col:
            if st.button(question, key=f"suggested_q_{i}"):
                st.session_state.pending_question = question
    st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_question = st.chat_input("Ask a question about your documents...")

# If a suggested question was clicked, use that instead
if st.session_state.pending_question:
    user_question = st.session_state.pending_question
    st.session_state.pending_question = None

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
                results = st.session_state.vectorstore.similarity_search(user_question, k=5)

            if not results:
                response_text = "I couldn't find enough information about this in the uploaded documents."
                st.write(response_text)
            else:
                with st.spinner("Generating answer..."):
                    try:
                        previous_history = st.session_state.messages[:-1]
                        answer = generate_answer(user_question, results, chat_history=previous_history)
                    except ValueError as e:
                        answer = f"⚠️ {str(e)}"

                st.write(answer)

                with st.expander(f"📚 View Sources ({len(results)})"):
                    seen_pages = set()

                    for res in results:
                        source = res.metadata.get("source")
                        page = res.metadata.get("page")
                        page_key = (source, page)

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