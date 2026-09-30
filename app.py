import streamlit as st

from src.pdf_loader import load_multiple_pdfs
from src.text_splitter import chunk_pages

from src.vector_store import (
    build_vectorstore,
    save_vectorstore,
    load_vectorstore,
    manifest_matches,
)

from src.rag_pipeline import (
    generate_answer,
    generate_study_answer
)

from src.summarizer import (
    summarize_document,
    generate_suggested_questions
)

from src.unit_detector import get_available_units, get_unit_text
from src.quiz_generator import generate_quiz

from src.viva_generator import (
    generate_viva_questions,
    evaluate_viva_answer
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI StudyMate",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================
if "pages_data" not in st.session_state:
    st.session_state.pages_data = None
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

# Quiz state
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None
# Viva state
if "viva_data" not in st.session_state:
    st.session_state.viva_data = None

if "viva_current_question" not in st.session_state:
    st.session_state.viva_current_question = 0

if "viva_answer" not in st.session_state:
    st.session_state.viva_answer = ""

if "viva_result" not in st.session_state:
    st.session_state.viva_result = None

if "viva_source_text" not in st.session_state:
    st.session_state.viva_source_text = ""


# =========================================================
# SIDEBAR
# =========================================================

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

        current_file_names = [
            f.name for f in uploaded_files
        ]

        # -------------------------------------------------
        # CASE 1: Already processed in this session
        # -------------------------------------------------

        if (
            tuple(current_file_names)
            == st.session_state.processed_files
        ):

            st.info(
                f"✅ {len(uploaded_files)} file(s) already processed"
            )


        # -------------------------------------------------
        # CASE 2: Existing vectorstore matches files
        # -------------------------------------------------

        elif manifest_matches(current_file_names):

            with st.spinner(
                "Loading saved vectorstore from disk..."
            ):

                st.session_state.vectorstore = (
                    load_vectorstore()
                )

                st.session_state.processed_files = (
                    tuple(current_file_names)
                )

            st.success(
                f"✅ Loaded existing vectorstore for "
                f"{len(uploaded_files)} file(s) from disk"
            )


        # -------------------------------------------------
        # CASE 3: New / different files
        # -------------------------------------------------

        else:

            with st.spinner("Reading PDFs..."):

                pages_data = load_multiple_pdfs(
                    uploaded_files
                )
            st.session_state.pages_data = pages_data

            with st.spinner(
                "Splitting text into chunks..."
            ):

                chunks = chunk_pages(
                    pages_data
                ) 

            with st.spinner(
                "Generating embeddings and building "
                "vectorstore... (this may take a minute)"
            ):

                vectorstore = build_vectorstore(
                    chunks
                )

                save_vectorstore(
                    vectorstore,
                    current_file_names
                )

            st.session_state.vectorstore = vectorstore

            st.session_state.processed_files = (
                tuple(current_file_names)
            )

            # New documents = reset quiz
            st.session_state.quiz_data = None
            st.session_state.quiz_submitted = False
            st.session_state.quiz_score = None

            st.success(
                f"✅ {len(uploaded_files)} file(s) processed — "
                f"{len(chunks)} chunks indexed"
            )


        # -------------------------------------------------
        # SHOW UPLOADED FILES
        # -------------------------------------------------

        for file in uploaded_files:

            st.write(
                f"- {file.name}"
            )

        st.divider()


        # =================================================
        # SUMMARIZE DOCUMENTS
        # =================================================

        if st.button("📝 Summarize Document(s)"):

            with st.spinner(
                "Reading documents for summarization..."
            ):

                pages_data = load_multiple_pdfs(
                    uploaded_files
                )

                chunks = chunk_pages(
                    pages_data
                )

            progress_bar = st.progress(
                0,
                text="Summarizing sections..."
            )

            def update_progress(
                current,
                total
            ):

                progress_bar.progress(
                    current / total,
                    text=(
                        f"Summarizing section "
                        f"{current} of {total}..."
                    )
                )

            with st.spinner(
                "Combining into final summary..."
            ):

                summary = summarize_document(
                    chunks,
                    progress_callback=update_progress
                )

            progress_bar.empty()

            st.session_state.document_summary = (
                summary
            )


        # =================================================
        # SUGGESTED QUESTIONS
        # =================================================

        if st.button("💡 Suggest Questions"):

            with st.spinner(
                "Generating suggested questions..."
            ):

                sample_pages = load_multiple_pdfs(
                    uploaded_files
                )

                sample_chunks = chunk_pages(
                    sample_pages
                )

                total = len(sample_chunks)

                if total <= 8:

                    picked = sample_chunks

                else:

                    step = max(
                        1,
                        total // 8
                    )

                    picked = [
                        sample_chunks[i]
                        for i in range(
                            0,
                            total,
                            step
                        )
                    ][:8]

                sample_text = "\n\n".join(
                    c["text"]
                    for c in picked
                )

                questions = (
                    generate_suggested_questions(
                        sample_text
                    )
                )

                st.session_state.suggested_questions = (
                    questions
                )


    # -----------------------------------------------------
    # NO DOCUMENTS UPLOADED
    # -----------------------------------------------------

    else:

        st.info(
            "No documents uploaded yet"
        )

        st.session_state.vectorstore = None

        st.session_state.processed_files = None


# =========================================================
# MAIN AREA
# =========================================================

st.title(
    "💬 Chat with your Documents"
)

st.caption(
    "Upload PDFs from the sidebar, then ask questions about them."
)


# =========================================================
# STUDY MODE
# =========================================================

study_mode = st.selectbox(
    "🎓 Study Mode",
    [
        "Explain Simply",
        "Quick Revision",
        "Important Topics",
        "Exam Answer - 2 Marks",
        "Exam Answer - 5 Marks",
        "Exam Answer - 10 Marks"
    ],
    help=(
        "Choose how AI StudyMate should present "
        "the answer."
    )
)


# =========================================================
# DOCUMENT SUMMARY
# =========================================================

if st.session_state.document_summary:

    with st.expander(
        "📝 Document Summary",
        expanded=True
    ):

        st.markdown(
            st.session_state.document_summary
        )

    st.divider()


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

if st.session_state.suggested_questions:

    st.markdown(
        "**💡 Try asking:**"
    )

    cols = st.columns(2)

    for i, question in enumerate(
        st.session_state.suggested_questions
    ):

        col = cols[i % 2]

        with col:

            if st.button(
                question,
                key=f"suggested_q_{i}"
            ):

                st.session_state.pending_question = (
                    question
                )

    st.divider()
# =========================================================
# VIVA MODE
# =========================================================

st.subheader("🎤 Viva Mode")

st.caption(
    "Practice viva questions generated from your uploaded study material."
)


if st.session_state.vectorstore is not None:

    pages_data = st.session_state.get("pages_data")

    if pages_data:

        # ---------------------------------------------
        # VIVA SETTINGS
        # ---------------------------------------------

        viva_question_count = st.selectbox(
            "🔢 Number of Viva Questions",
            [5, 10, 15, 20],
            index=0,
            key="selected_viva_question_count"
        )


        # ---------------------------------------------
        # START VIVA
        # ---------------------------------------------

        if st.session_state.viva_data is None:

            if st.button(
                "🎤 Start Viva",
                type="primary"
            ):

                with st.spinner(
                    "Preparing your viva questions..."
                ):

                    try:

                        # Build study material
                        source_parts = []

                        for page in pages_data:

                            text = page.get(
                                "text",
                                ""
                            ).strip()

                            if text:
                                source_parts.append(text)


                        source_text = (
                            "\n\n".join(
                                source_parts
                            )
                        ).strip()


                        if len(source_text) < 100:

                            st.warning(
                                "⚠️ Not enough study material "
                                "was found for the viva."
                            )

                        else:

                            viva_questions = (
                                generate_viva_questions(
                                    source_text,
                                    num_questions=viva_question_count
                                )
                            )


                            if not viva_questions:

                                st.warning(
                                    "⚠️ Could not generate "
                                    "viva questions."
                                )

                            else:

                                st.session_state.viva_data = (
                                    viva_questions
                                )

                                st.session_state.viva_current_question = 0

                                st.session_state.viva_answer = ""

                                st.session_state.viva_result = None

                                st.session_state.viva_source_text = (
                                    source_text
                                )

                                st.rerun()


                    except ValueError as e:

                        st.error(
                            f"⚠️ {str(e)}"
                        )

                    except Exception as e:

                        st.error(
                            f"⚠️ Viva generation failed: "
                            f"{str(e)}"
                        )


    else:

        st.warning(
            "⚠️ Please re-upload the PDF before starting Viva Mode."
        )


else:

    st.info(
        "📄 Upload a PDF to start Viva Mode."
    )


# ---------------------------------------------------------
# DISPLAY VIVA
# ---------------------------------------------------------

if st.session_state.viva_data:

    viva_questions = st.session_state.viva_data

    current_index = (
        st.session_state.viva_current_question
    )

    current_question = (
        viva_questions[current_index]
    )


    st.divider()

    st.markdown(
        f"### 🎤 Viva Question "
        f"{current_index + 1} / {len(viva_questions)}"
    )


    # Difficulty
    difficulty = current_question.get(
        "difficulty",
        "Medium"
    )

    st.caption(
        f"Difficulty: **{difficulty}**"
    )


    # Question
    st.markdown(
        f"### ❓ {current_question['question']}"
    )


    # -----------------------------------------------------
    # ANSWER INPUT
    # -----------------------------------------------------

    if st.session_state.viva_result is None:

        answer = st.text_area(
            "✍️ Your Answer",
            value=st.session_state.viva_answer,
            height=150,
            placeholder="Type your answer here..."
        )

        st.session_state.viva_answer = answer


        # -------------------------------------------------
        # EVALUATE ANSWER
        # -------------------------------------------------

        if st.button(
            "✅ Submit Answer",
            type="primary"
        ):

            if not answer.strip():

                st.warning(
                    "⚠️ Please write an answer first."
                )

            else:

                with st.spinner(
                    "Evaluating your answer..."
                ):

                    try:

                        result = evaluate_viva_answer(
                            question=current_question["question"],
                            student_answer=answer,
                            source_text=st.session_state.viva_source_text
                        )

                        st.session_state.viva_result = (
                            result
                        )

                        st.rerun()

                    except ValueError as e:

                        st.error(
                            f"⚠️ {str(e)}"
                        )

                    except Exception as e:

                        st.error(
                            f"⚠️ Evaluation failed: "
                            f"{str(e)}"
                        )


    # -----------------------------------------------------
    # DISPLAY RESULT
    # -----------------------------------------------------

    else:

        result = st.session_state.viva_result

        score = result.get(
            "score",
            0
        )

        evaluation = result.get(
            "evaluation",
            "Unable to evaluate"
        )

        feedback = result.get(
            "feedback",
            ""
        )

        suggested_answer = result.get(
            "suggested_answer",
            ""
        )


        st.markdown("### 📊 Evaluation")


        if score == 3:

            st.success(
                f"✅ {evaluation} — {score}/3"
            )

        elif score == 2:

            st.info(
                f"🟡 {evaluation} — {score}/3"
            )

        elif score == 1:

            st.warning(
                f"🟠 {evaluation} — {score}/3"
            )

        else:

            st.error(
                f"❌ {evaluation} — {score}/3"
            )


        st.markdown("**💬 Feedback**")

        st.write(
            feedback
        )


        if suggested_answer:

            with st.expander(
                "📖 Suggested Answer",
                expanded=True
            ):

                st.write(
                    suggested_answer
                )


        st.divider()


        # -------------------------------------------------
        # NEXT QUESTION
        # -------------------------------------------------

        if current_index < len(viva_questions) - 1:

            if st.button(
                "➡️ Next Question",
                type="primary"
            ):

                st.session_state.viva_current_question += 1

                st.session_state.viva_answer = ""

                st.session_state.viva_result = None

                st.rerun()


        else:

            st.success(
                "🎉 Viva completed!"
            )

            if st.button(
                "🔄 Start New Viva"
            ):

                st.session_state.viva_data = None

                st.session_state.viva_current_question = 0

                st.session_state.viva_answer = ""

                st.session_state.viva_result = None

                st.session_state.viva_source_text = ""

                st.rerun()
# =========================================================
# QUIZ MODE
# =========================================================

st.subheader("🎯 Quiz Mode")

st.caption(
    "Test your understanding using MCQs generated "
    "from your uploaded study material."
)


# ---------------------------------------------------------
# QUIZ SETTINGS
# ---------------------------------------------------------

if st.session_state.vectorstore is not None:

    pages_data = st.session_state.get(
        "pages_data"
    )

    # Detect available units
    if pages_data:

        available_units = get_available_units(
            pages_data
        )

    else:

        available_units = []


    # If no units detected, use Full Document
    if not available_units:

        quiz_options = [
            "Full Document"
        ]

    else:

        quiz_options = [
            "Full Syllabus"
        ] + available_units


    # -----------------------------------------------------
    # QUIZ TYPE
    # -----------------------------------------------------

    quiz_type = st.selectbox(
        "📚 Quiz Type",
        quiz_options,
        key="selected_quiz_type"
    )


    # -----------------------------------------------------
    # QUESTION COUNT
    # -----------------------------------------------------

    question_count = st.selectbox(
        "🔢 Number of Questions",
        [20, 25, 30, 40, 50],
        index=0,
        key="selected_question_count"
    )


    # -----------------------------------------------------
    # START QUIZ
    # -----------------------------------------------------

    if st.session_state.quiz_data is None:

        if st.button(
            "🚀 Start Quiz",
            type="primary"
        ):

            with st.spinner(
                "Preparing your quiz..."
            ):

                try:

                    # -------------------------------------
                    # SELECT SOURCE MATERIAL
                    # -------------------------------------

                    if quiz_type == "Full Syllabus":

                        # Combine all PDF pages
                        source_pages = pages_data

                    elif quiz_type == "Full Document":

                        source_pages = pages_data

                    else:

                        # Get selected unit text
                        unit_text = get_unit_text(
                            pages_data,
                            quiz_type
                        )

                        source_pages = [
                            {
                                "text": unit_text
                            }
                        ]


                    # -------------------------------------
                    # BUILD SOURCE TEXT
                    # -------------------------------------

                    source_parts = []

                    for page in source_pages:

                        text = page.get(
                            "text",
                            ""
                        ).strip()

                        if text:

                            source_parts.append(
                                text
                            )


                    source_text = (
                        "\n\n".join(
                            source_parts
                        )
                    ).strip()


                    # -------------------------------------
                    # CHECK SOURCE
                    # -------------------------------------

                    if len(source_text) < 100:

                        st.warning(
                            "⚠️ Not enough study material "
                            "was found for this quiz."
                        )

                    else:

                        # ---------------------------------
                        # GENERATE QUIZ
                        # ---------------------------------

                        quiz = generate_quiz(
                            source_text,
                            num_questions=question_count
                        )


                        if not quiz:

                            st.warning(
                                "⚠️ Could not generate "
                                "the quiz. Please try again."
                            )

                        else:

                            st.session_state.quiz_data = quiz

                            st.session_state.quiz_title = (
                                quiz_type
                            )

                            st.session_state.quiz_submitted = (
                                False
                            )

                            st.session_state.quiz_score = (
                                None
                            )

                            st.rerun()


                except ValueError as e:

                    st.error(
                        f"⚠️ {str(e)}"
                    )

                except Exception as e:

                    st.error(
                        f"⚠️ Quiz generation failed: "
                        f"{str(e)}"
                    )


else:

    st.info(
        "📄 Upload a PDF to start a quiz."
    )


# ---------------------------------------------------------
# DISPLAY QUIZ
# ---------------------------------------------------------

if st.session_state.quiz_data:

    quiz = st.session_state.quiz_data

    quiz_title = (
        st.session_state.get(
            "quiz_title",
            "Quiz"
        )
    )


    st.divider()

    st.markdown(
        f"### 📝 {quiz_title} Quiz — "
        f"{len(quiz)} Questions"
    )


    # -----------------------------------------------------
    # QUESTIONS
    # -----------------------------------------------------

    if not st.session_state.quiz_submitted:

        for i, question in enumerate(
            quiz
        ):

            st.markdown(
                f"**Q{i + 1}. "
                f"{question['question']}**"
            )

            st.radio(
                "Choose your answer:",
                question["options"],
                key=f"quiz_answer_{i}",
                index=None
            )

            st.divider()


        # -------------------------------------------------
        # SUBMIT QUIZ
        # -------------------------------------------------

        if st.button(
            "✅ Submit Quiz",
            type="primary"
        ):

            unanswered = []

            for i in range(
                len(quiz)
            ):

                selected = (
                    st.session_state.get(
                        f"quiz_answer_{i}"
                    )
                )

                if selected is None:

                    unanswered.append(
                        i + 1
                    )


            if unanswered:

                st.warning(
                    "⚠️ Please answer all "
                    "questions before submitting. "
                    f"Unanswered: "
                    f"{', '.join(map(str, unanswered))}"
                )

            else:

                score = 0


                for i, question in enumerate(
                    quiz
                ):

                    selected = (
                        st.session_state[
                            f"quiz_answer_{i}"
                        ]
                    )

                    correct_index = (
                        question[
                            "correct_answer"
                        ]
                    )

                    correct_option = (
                        question[
                            "options"
                        ][correct_index]
                    )


                    if selected == correct_option:

                        score += 1


                st.session_state.quiz_score = (
                    score
                )

                st.session_state.quiz_submitted = (
                    True
                )

                st.rerun()


    # -----------------------------------------------------
    # QUIZ RESULT
    # -----------------------------------------------------

    else:

        score = (
            st.session_state.quiz_score
        )


        st.success(
            f"🎉 Quiz Completed! "
            f"Your Score: "
            f"**{score} / {len(quiz)}**"
        )


        percentage = (
            score / len(quiz)
        ) * 100


        st.progress(
            percentage / 100
        )


        st.write(
            f"**Score:** "
            f"{percentage:.0f}%"
        )


        st.divider()


        for i, question in enumerate(
            quiz
        ):

            selected = (
                st.session_state.get(
                    f"quiz_answer_{i}"
                )
            )


            correct_index = (
                question[
                    "correct_answer"
                ]
            )


            correct_option = (
                question[
                    "options"
                ][correct_index]
            )


            st.markdown(
                f"### Q{i + 1}. "
                f"{question['question']}"
            )


            if selected == correct_option:

                st.success(
                    f"✅ Your answer: "
                    f"{selected}"
                )

            else:

                st.error(
                    f"❌ Your answer: "
                    f"{selected}"
                )

                st.info(
                    f"✅ Correct answer: "
                    f"{correct_option}"
                )


            st.markdown(
                f"**Explanation:** "
                f"{question['explanation']}"
            )


            st.divider()


        # -------------------------------------------------
        # START NEW QUIZ
        # -------------------------------------------------

        if st.button(
            "🔄 Start New Quiz"
        ):

            st.session_state.quiz_data = None

            st.session_state.quiz_submitted = (
                False
            )

            st.session_state.quiz_score = None

            st.session_state.quiz_title = None

            st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask a question about your documents..."
)


# =========================================================
# SUGGESTED QUESTION HANDLING
# =========================================================

if st.session_state.pending_question:

    user_question = (
        st.session_state.pending_question
    )

    st.session_state.pending_question = None


# =========================================================
# QUESTION PROCESSING
# =========================================================

if user_question:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.write(
            user_question
        )


    # -----------------------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        # =================================================
        # CASE 1: NO VECTORSTORE
        # =================================================

        if st.session_state.vectorstore is None:

            response_text = (
                "⚠️ Please upload at least one PDF "
                "before asking questions."
            )

            st.write(
                response_text
            )


        # =================================================
        # CASE 2: VECTORSTORE AVAILABLE
        # =================================================

        else:

            # ---------------------------------------------
            # SEARCH RELEVANT CHUNKS
            # ---------------------------------------------

            with st.spinner(
                "Searching your documents..."
            ):

                results = (
                    st.session_state.vectorstore
                    .similarity_search(
                        user_question,
                        k=3
                    )
                )


            # ---------------------------------------------
            # CASE 2A: NO RESULTS
            # ---------------------------------------------

            if not results:

                response_text = (
                    "I couldn't find enough information "
                    "about this in the uploaded documents."
                )

                st.write(
                    response_text
                )


            # ---------------------------------------------
            # CASE 2B: RESULTS FOUND
            # ---------------------------------------------

            else:

                with st.spinner(
                    "Preparing your answer..."
                ):

                    try:

                        previous_history = (
                            st.session_state.messages[:-1]
                        )

                        answer = generate_study_answer(
                            user_question,
                            results,
                            study_mode,
                            chat_history=previous_history
                        )

                    except ValueError as e:

                        answer = (
                            f"⚠️ {str(e)}"
                        )


                # -----------------------------------------
                # DISPLAY ANSWER
                # -----------------------------------------

                st.write(
                    answer
                )


                # -----------------------------------------
                # DISPLAY SOURCES
                # -----------------------------------------

                with st.expander(
                    f"📚 View Sources ({len(results)})"
                ):

                    seen_pages = set()

                    for res in results:

                        source = res.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = res.metadata.get(
                            "page",
                            "Unknown"
                        )

                        page_key = (
                            source,
                            page
                        )

                        # Avoid duplicate pages
                        if page_key in seen_pages:
                            continue

                        seen_pages.add(
                            page_key
                        )


                        # ---------------------------------
                        # CLEAN SOURCE TEXT
                        # ---------------------------------

                        text_snippet = (
                            res.page_content.strip()
                        )

                        # Remove common extraction artifacts
                        text_snippet = (
                            text_snippet
                            .replace("svg", "")
                            .replace("SVG", "")
                            .replace("¢", "")
                        )

                        # Normalize spaces/newlines
                        text_snippet = (
                            " ".join(
                                text_snippet.split()
                            )
                        )


                        # ---------------------------------
                        # LIMIT PREVIEW LENGTH
                        # ---------------------------------

                        if len(text_snippet) > 220:

                            text_snippet = (
                                text_snippet[:220]
                                .rsplit(" ", 1)[0]
                                + "..."
                            )


                        # ---------------------------------
                        # SOURCE DISPLAY
                        # ---------------------------------

                        st.markdown(
                            f"**📄 {source}**"
                        )

                        st.markdown(
                            f"📌 **Page {page}**"
                        )

                        if text_snippet:

                            st.caption(
                                text_snippet
                            )

                        st.divider()


                # -----------------------------------------
                # FINAL RESPONSE TEXT
                # -----------------------------------------

                response_text = answer


    # -----------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text
        }
    )