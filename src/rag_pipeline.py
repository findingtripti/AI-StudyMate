import os
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add it to the .env file."
        )

    return Groq(api_key=api_key)


# ============================================================
# CHAT HISTORY
# ============================================================

def format_chat_history(chat_history, max_messages=6):

    if not chat_history:
        return ""

    recent_messages = chat_history[-max_messages:]

    history = []

    for message in recent_messages:

        role = message.get("role", "")
        content = message.get("content", "").strip()

        if not content:
            continue

        if role == "user":
            label = "Student"
        elif role == "assistant":
            label = "AI StudyMate"
        else:
            label = role.capitalize()

        history.append(
            f"{label}: {content}"
        )

    return "\n".join(history)


# ============================================================
# NORMAL RAG PROMPT
# ============================================================

def build_prompt(
    question,
    retrieved_chunks,
    chat_history=None
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        text = chunk.page_content.strip()

        source = chunk.metadata.get(
            "source",
            "Unknown"
        )

        page = chunk.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"[Source {i}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    history = format_chat_history(
        chat_history
    )

    history_section = ""

    if history:
        history_section = f"""
PREVIOUS CONVERSATION:
{history}
"""

    return f"""
You are AI StudyMate, an intelligent study assistant.

Answer the student's question using ONLY the supplied study material.

Rules:
- Be accurate.
- Do not invent information.
- Explain difficult concepts simply.
- Stay focused on the question.
- Use headings and bullet points when useful.
- If the answer is not present in the documents, say so clearly.

{history_section}

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""


# ============================================================
# NORMAL ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    chat_history=None
):

    if not retrieved_chunks:
        return (
            "I couldn't find enough information "
            "about this in the uploaded documents."
        )

    client = get_groq_client()

    prompt = build_prompt(
        question,
        retrieved_chunks,
        chat_history
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1200
    )

    answer = response.choices[0].message.content

    if not answer:
        return (
            "I couldn't generate an answer. "
            "Please try again."
        )

    return answer.strip()


# ============================================================
# QUICK REVISION - TEXT CLEANING
# ============================================================

def _clean_pdf_text(text):

    if not text:
        return ""

    text = str(text)

    # Normalize line breaks
    text = text.replace("\r", "\n")

    # Common PDF/OCR bullet artifacts
    text = text.replace("¢", " ")
    text = text.replace("°", " ")
    text = text.replace("•", " ")
    text = text.replace("▪", " ")
    text = text.replace("◦", " ")
    text = text.replace("●", " ")

    # Remove markdown bullets
    text = re.sub(
        r"(^|\n)\s*[-*]\s+",
        r"\1",
        text
    )

    # Fix words joined after punctuation
    text = re.sub(
        r"([.!?])([A-Z])",
        r"\1 \2",
        text
    )

    text = re.sub(
        r"(\))([A-Z])",
        r"\1 \2",
        text
    )

    # Fix common PDF missing spaces
    text = re.sub(
        r"([a-z])([A-Z][a-z])",
        r"\1 \2",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n+",
        "\n",
        text
    )

    return text.strip()


# ============================================================
# QUICK REVISION - SPLIT PDF CONTENT
# ============================================================

def _split_revision_text(text):

    text = _clean_pdf_text(text)

    if not text:
        return []

    # --------------------------------------------------------
    # Remove obvious navigation / TOC material
    # --------------------------------------------------------

    toc_patterns = [
        r"Stochastic Optimization Generalization in neural networks",
        r"Spatial IV",
        r"Transformer Networks",
        r"Recurrent networks",
        r"Recurrent Neural Network Language Models",
        r"Word-Level RNNs",
        r"Deep Reinforcement Learning",
        r"Computational & Artificial Neuroscience",
        r"Computational and Artificial Neuroscience",
    ]

    for pattern in toc_patterns:

        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )

    # --------------------------------------------------------
    # Turn PDF bullet-like "e" artifacts into boundaries
    # Example:
    # e There are many local minima
    # e The shape is complex
    # --------------------------------------------------------

    text = re.sub(
        r"\s+e\s+(?=[A-Z])",
        "\n",
        text
    )

    # --------------------------------------------------------
    # Split known section headings
    # --------------------------------------------------------

    heading_patterns = [
        r"How Optimizers Handle Non-Convexity",
        r"What Is Stochastic Optimization\??",
        r"What Does Non-Convex Optimization Mean\??",
        r"Why Is Optimization Important\??",
    ]

    for pattern in heading_patterns:

        text = re.sub(
            pattern,
            "\n",
            text,
            flags=re.IGNORECASE
        )

    # --------------------------------------------------------
    # Split normal sentences
    # --------------------------------------------------------

    text = re.sub(
        r"(?<=[.!?])\s+(?=[A-Z])",
        "\n",
        text
    )

    # Handle punctuation immediately followed by capital letter
    text = re.sub(
        r"([.!?])(?=[A-Z])",
        r"\1\n",
        text
    )

    # --------------------------------------------------------
    # Split common PDF bullet phrases
    # --------------------------------------------------------

    bullet_starts = [
        "It’s",
        "It's",
        "There are",
        "The shape",
        "They smooth",
        "They damp",
        "Reduces",
        "Helps",
        "To do this",
    ]

    for phrase in bullet_starts:

        text = re.sub(
            rf"\s+(?={re.escape(phrase)}\b)",
            "\n",
            text
        )

    # --------------------------------------------------------
    # Create clean lines
    # --------------------------------------------------------

    pieces = []

    for part in text.split("\n"):

        part = _clean_pdf_text(part)

        if not part:
            continue

        pieces.append(part)

    return pieces


# ============================================================
# QUICK REVISION - VALIDATION
# ============================================================

def _is_good_revision_point(text):

    text = _clean_pdf_text(text)

    if not text:
        return False

    lower = text.lower()

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if len(text) < 45:
        return False

    if len(text) > 280:
        return False

    # --------------------------------------------------------
    # Incomplete sentence
    # --------------------------------------------------------

    if not text.endswith(
        (".", "!", "?", ":")
    ):
        return False

    bad_endings = [
        " parameters.",
        " weights.",
        " biases.",
        " because.",
        " where.",
        " which.",
        " and.",
        " or.",
        " the.",
        " a.",
        " an.",
        " of.",
        " to.",
        " with.",
        " from.",
        " in.",
        " on.",
    ]

    if lower.endswith(
        tuple(bad_endings)
    ):
        return False

    # --------------------------------------------------------
    # Garbage / UI text
    # --------------------------------------------------------

    garbage = [
        "[svg]",
        "svg",
        "localhost",
        "view sources",
        "<svg",
        "</svg>",
        "open",
        "revision",
        "quick revision",
    ]

    if any(
        item in lower
        for item in garbage
    ):
        return False

    # --------------------------------------------------------
    # Table of contents
    # --------------------------------------------------------

    toc_words = [
        "spatial iv",
        "transformer networks",
        "recurrent networks",
        "word-level rnns",
        "deep reinforcement learning",
        "computational & artificial neuroscience",
        "computational and artificial neuroscience",
        "recurrent neural network language models",
    ]

    if any(
        item in lower
        for item in toc_words
    ):
        return False

    # --------------------------------------------------------
    # Too many separators
    # --------------------------------------------------------

    if text.count("|") >= 2:
        return False

    if text.count("—") >= 3:
        return False

    if text.count("-") >= 4:
        return False

    # --------------------------------------------------------
    # Reject headings
    # --------------------------------------------------------

    headings = [
        "optimization in deep learning",
        "how optimizers handle non-convexity",
        "what is stochastic optimization",
        "what does non-convex optimization mean",
        "flow diagram",
        "contents",
        "table of contents",
    ]

    if lower.rstrip(":") in headings:
        return False

    return True


# ============================================================
# QUICK REVISION - EXTRACT POINTS
# ============================================================

def _extract_revision_points(
    question,
    retrieved_chunks,
    max_points=5
):

    candidates = []

    # --------------------------------------------------------
    # Collect sentences from retrieved chunks
    # --------------------------------------------------------

    for chunk in retrieved_chunks:

        raw_text = chunk.page_content

        pieces = _split_revision_text(
            raw_text
        )

        for piece in pieces:

            piece = _clean_pdf_text(
                piece
            )

            if _is_good_revision_point(
                piece
            ):
                candidates.append(
                    piece
                )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = []
    seen = set()

    for point in candidates:

        normalized = re.sub(
            r"[^a-z0-9 ]",
            "",
            point.lower()
        )

        normalized = re.sub(
            r"\s+",
            " ",
            normalized
        ).strip()

        if normalized in seen:
            continue

        seen.add(normalized)

        unique.append(point)

    # --------------------------------------------------------
    # Relevance scoring
    # --------------------------------------------------------

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    stop_words = {
        "what",
        "why",
        "how",
        "does",
        "this",
        "that",
        "with",
        "from",
        "about",
        "into",
        "where",
        "when",
        "which",
        "the",
        "and",
        "for",
        "are",
    }

    question_words -= stop_words

    important_keywords = [
        "optimization",
        "loss function",
        "gradient",
        "stochastic optimization",
        "gradient descent",
        "optimizer",
        "optimizers",
        "learning rate",
        "parameters",
        "weights",
        "biases",
        "non-convex",
        "deep learning",
        "momentum",
        "adam",
        "rmsprop",
    ]

    def score(point):

        lower = point.lower()

        words = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                lower
            )
        )

        score_value = len(
            words.intersection(
                question_words
            )
        )

        for keyword in important_keywords:

            if keyword in lower:
                score_value += 3

        # Prefer readable medium-sized points
        if 70 <= len(point) <= 220:
            score_value += 2

        return score_value

    unique.sort(
        key=score,
        reverse=True
    )

    return unique[:max_points]


# ============================================================
# QUICK REVISION - FLOW
# ============================================================

def _create_flow_from_context(
    question,
    retrieved_chunks
):

    question_lower = question.lower()

    if (
        "optimization" in question_lower
        or "optimizer" in question_lower
        or "gradient" in question_lower
    ):

        return (
            "Adjust model parameters "
            "(weights and biases)\n"
            "      ↓\n"
            "Calculate / minimize the loss\n"
            "      ↓\n"
            "Update parameters using gradients\n"
            "      ↓\n"
            "Make better predictions"
        )

    if (
        "classification" in question_lower
    ):

        return (
            "Input data\n"
            "      ↓\n"
            "Extract features\n"
            "      ↓\n"
            "Apply classification model\n"
            "      ↓\n"
            "Predict class"
        )

    if (
        "training" in question_lower
        or "train" in question_lower
    ):

        return (
            "Input training data\n"
            "      ↓\n"
            "Model makes prediction\n"
            "      ↓\n"
            "Calculate loss\n"
            "      ↓\n"
            "Update model parameters\n"
            "      ↓\n"
            "Repeat until performance improves"
        )

    return (
        "Input study information\n"
        "      ↓\n"
        "Process the information\n"
        "      ↓\n"
        "Apply the relevant concept\n"
        "      ↓\n"
        "Obtain the result"
    )


# ============================================================
# QUICK REVISION
# ============================================================

def generate_quick_revision(
    question,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "I couldn't find enough information "
            "about this in the uploaded documents."
        )

    points = _extract_revision_points(
        question,
        retrieved_chunks,
        max_points=5
    )

    if not points:

        return (
            "I couldn't find enough clearly "
            "relevant information in the "
            "uploaded documents."
        )

    flow = _create_flow_from_context(
        question,
        retrieved_chunks
    )

    answer_parts = [
        "### Quick Revision"
    ]

    for point in points:

        answer_parts.append(
            f"- {point}"
        )

    answer_parts.append(
        "\n### Flow Diagram"
    )

    answer_parts.append(
        "```text\n"
        + flow
        + "\n```"
    )

    return "\n".join(
        answer_parts
    )


# ============================================================
# STUDY MODE PROMPT
# ============================================================

def _build_study_prompt(
    question,
    retrieved_chunks,
    study_mode,
    chat_history=None
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        text = chunk.page_content.strip()

        page = chunk.metadata.get(
            "page",
            "Unknown"
        )

        context_parts.append(
            f"[Source {i} | Page {page}]\n"
            f"{text}"
        )

    context = "\n\n".join(
        context_parts
    )

    history = format_chat_history(
        chat_history
    )

    history_section = ""

    if history:
        history_section = f"""
PREVIOUS CHAT:
{history}
"""

    if study_mode == "Explain Simply":

        instruction = """
Explain the topic in simple student-friendly language.
Give a short definition, explanation and example if useful.
"""

    elif study_mode == "Important Topics":

        instruction = """
Identify the important concepts related to the question.
Use clear headings and bullet points.
"""

    elif study_mode == "Exam Answer - 2 Marks":

        instruction = """
Write a concise 2-mark exam answer.
Give the definition and one important point.
"""

    elif study_mode == "Exam Answer - 5 Marks":

        instruction = """
Write a 5-mark exam-ready answer.

Structure:
1. Definition
2. Explanation
3. Important points
4. Example/application if relevant
5. Conclusion
"""

    elif study_mode == "Exam Answer - 10 Marks":

        instruction = """
Write a detailed 10-mark exam-ready answer.

Structure:
1. Introduction
2. Definition
3. Detailed explanation
4. Working/mechanism
5. Important techniques or components
6. Example/application
7. Conclusion
"""

    else:

        instruction = """
Give a clear study-oriented explanation.
"""

    return f"""
You are AI StudyMate.

Answer the student's question using ONLY the study material.

{instruction}

Rules:
- Do not invent information.
- Do not add unrelated topics.
- Do not reproduce table-of-contents text.
- Use clean Markdown.
- Write complete sentences.
- Keep the answer relevant to the question.

{history_section}

STUDY MATERIAL:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""


# ============================================================
# STUDY ANSWER
# ============================================================

def generate_study_answer(
    question,
    retrieved_chunks,
    study_mode,
    chat_history=None
):

    if not retrieved_chunks:

        return (
            "I couldn't find enough information "
            "about this in the uploaded documents."
        )

    # Quick Revision is intentionally local.
    # It does NOT make an additional Groq call.
    if study_mode == "Quick Revision":

        return generate_quick_revision(
            question,
            retrieved_chunks
        )

    client = get_groq_client()

    prompt = _build_study_prompt(
        question,
        retrieved_chunks,
        study_mode,
        chat_history
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1600
    )

    answer = response.choices[0].message.content

    if not answer:

        return (
            "I couldn't generate an answer. "
            "Please try again."
        )

    return answer.strip()