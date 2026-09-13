import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

_groq_client = None


def get_groq_client():
    """Returns a cached Groq client, initialized once."""
    global _groq_client

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "GROQ_API_KEY is missing or not set in .env file. "
                "Please add your Groq API key to the .env file."
            )
        _groq_client = Groq(api_key=api_key)

    return _groq_client


def build_prompt(question, retrieved_chunks):
    """
    Builds a prompt that instructs the LLM to answer ONLY from the
    provided context, and to clearly say when it can't find an answer.

    Args:
        question: The user's question (string)
        retrieved_chunks: List of LangChain Document objects (from similarity_search)

    Returns:
        A formatted prompt string
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "?")
        context_parts.append(
            f"[Source {i}: {source}, Page {page}]\n{chunk.page_content}"
        )

    context_text = "\n\n".join(context_parts)

    prompt = f"""You are AI StudyMate, a helpful study assistant for students.

Answer the student's question using ONLY the information provided in the CONTEXT below.

Rules you must follow:
1. Base your answer strictly on the given context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, say clearly: "I couldn't find enough information about this in the uploaded documents." Do not guess or make up an answer.
3. Keep the answer clear, well-structured, and easy for a student to understand.
4. Do not mention "Source 1", "Source 2" etc. in your answer text itself — the sources will be shown separately.
5.If the student's question specifies a particular format or length (e.g., "in 200 words", "in 5 marks", "in bullet points", "in one line"), follow that instruction as closely as possible while still staying grounded in the context.

CONTEXT:
{context_text}

QUESTION:
{question}

ANSWER:"""

    return prompt


def generate_answer(question, retrieved_chunks, model="openai/gpt-oss-20b"):
    """
    Generates an answer using Groq's LLM, based on retrieved context.

    Args:
        question: The user's question
        retrieved_chunks: List of LangChain Document objects
        model: Groq model name to use

    Returns:
        The generated answer as a string
    """
    if not retrieved_chunks:
        return "I couldn't find enough information about this in the uploaded documents."

    prompt = build_prompt(question, retrieved_chunks)
    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # low temperature = more focused, less "creative"/random answers
        max_tokens=4000
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        return "⚠️ The AI generated an empty response. Please try rephrasing your question or asking again."

    return answer