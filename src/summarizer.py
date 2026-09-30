import time
from groq import RateLimitError
from src.rag_pipeline import get_groq_client

MODEL = "openai/gpt-oss-20b"
CHUNKS_PER_GROUP = 15  # how many chunks go into one "mini summary" batch
SUMMARIES_PER_BATCH = 5  # how many partial summaries combined per reduce round


def _summarize_text_block(text_block, is_final=False):
    """
    Sends a block of text to the LLM and asks for a summary.
    Retries automatically if a rate limit is hit.
    """
    client = get_groq_client()

    if is_final:
        instruction = (
            "You are given several partial summaries of different sections of a study document. "
            "Combine them into ONE clear, well-organized final summary for a student. "
            "Use headings or bullet points where helpful. "
            "Do not repeat the same point multiple times. "
            "Keep it comprehensive but concise (aim for 300-500 words)."
        )
    else:
        instruction = (
            "Summarize the following section of a study document in a clear, concise way. "
            "Focus on the key concepts, definitions, and important points. "
            "Keep it factual and based only on the given text."
        )

    prompt = f"{instruction}\n\nTEXT:\n{text_block}\n\nSUMMARY:"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                raise


def summarize_document(chunks, progress_callback=None):
    """
    Summarizes a full document using a Map-Reduce approach:
    1. MAP: Split chunks into groups, summarize each group separately.
    2. REDUCE: Combine partial summaries in batches until one final summary remains.
    """
    if not chunks:
        return "No content available to summarize."

    # ---- MAP step ----
    groups = [
        chunks[i:i + CHUNKS_PER_GROUP]
        for i in range(0, len(chunks), CHUNKS_PER_GROUP)
    ]

    partial_summaries = []
    for i, group in enumerate(groups):
        group_text = "\n\n".join(chunk["text"] for chunk in group)
        summary = _summarize_text_block(group_text, is_final=False)
        partial_summaries.append(summary)

        if progress_callback:
            progress_callback(i + 1, len(groups))

        time.sleep(3)

    # ---- REDUCE step (hierarchical, to avoid oversized requests) ----
    current_level = partial_summaries
    while len(current_level) > 1:
        next_level = []
        batches = [
            current_level[i:i + SUMMARIES_PER_BATCH]
            for i in range(0, len(current_level), SUMMARIES_PER_BATCH)
        ]
        for batch in batches:
            batch_text = "\n\n---\n\n".join(batch)
            is_last_batch_round = (len(batches) == 1)
            combined = _summarize_text_block(batch_text, is_final=is_last_batch_round)
            next_level.append(combined)
            time.sleep(3)
        current_level = next_level

    return current_level[0]


def generate_suggested_questions(source_text, num_questions=5):
    """
    Generates a list of suggested study questions based on the document
    (or its summary). Returns a clean Python list of question strings.
    """
    client = get_groq_client()

    prompt = f"""Based on the following study material, generate exactly {num_questions} good study questions
that a student could ask to test or deepen their understanding of this material.

Rules:
- Questions should be answerable from the material itself.
- Make them varied: include at least one definition-type question, one "explain/how" question, and one comparison or application question if relevant.
- Output ONLY the questions, one per line, with no numbering, no bullets, no headings, and no extra commentary.
- Each line must be a single, complete question ending with a question mark.

MATERIAL:
{source_text}

QUESTIONS:"""

    response = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1500
            )
            break
        except RateLimitError as e:
            print(f"=== RATE LIMIT ERROR (attempt {attempt+1}) ===")
            print(str(e))
            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                return ["⚠️ Rate limit reached. Please wait a minute and try again."]
        except Exception as e:
            print(f"=== UNEXPECTED ERROR (attempt {attempt+1}) ===")
            print(str(e))
            return [f"⚠️ Error: {str(e)}"]

    if response is None:
        return ["⚠️ Could not generate questions. Please try again."]

    raw_output = response.choices[0].message.content.strip()

    print("=== RAW LLM OUTPUT FOR QUESTIONS ===")
    print(raw_output)
    print("=====================================")

    # Split into lines, clean up numbering/bullets, keep only lines that look like questions
    questions = []
    for line in raw_output.split("\n"):
        cleaned = line.strip().lstrip("0123456789.-•) ").strip()
        if cleaned and len(cleaned) > 10:
            questions.append(cleaned)

    if not questions:
        return ["⚠️ Could not generate questions. Please try again."]

    return questions[:num_questions]