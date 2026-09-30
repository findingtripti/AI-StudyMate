import json
import re
import time

from groq import RateLimitError

from src.rag_pipeline import get_groq_client


MODEL = "openai/gpt-oss-20b"

# Keep Groq requests within the available token limit
MAX_SOURCE_CHARS = 18000


def _clean_json_output(raw_output):
    """Remove markdown code fences if the model returns them."""

    if not raw_output:
        return ""

    text = raw_output.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    return text.strip()


def _limit_source_text(source_text):
    """Limit study material size before sending it to Groq."""

    if not source_text:
        return ""

    source_text = source_text.strip()

    if len(source_text) > MAX_SOURCE_CHARS:
        source_text = source_text[:MAX_SOURCE_CHARS]

    return source_text


def generate_viva_questions(source_text, num_questions=10):
    """
    Generate viva questions from the uploaded study material.
    """

    if not source_text or not source_text.strip():
        return []

    try:
        num_questions = int(num_questions)
    except (TypeError, ValueError):
        num_questions = 10

    num_questions = max(5, min(num_questions, 20))

    # Limit source text to avoid Groq TPM/request-size errors
    source_text = _limit_source_text(source_text)

    client = get_groq_client()

    prompt = f"""
You are an AI viva examiner for a college student.

Based ONLY on the study material provided below, generate exactly
{num_questions} viva questions.

Rules:
- Questions must be answerable from the provided material.
- Do not use outside knowledge.
- Keep questions suitable for a B.Tech CSE student.
- Mix different difficulty levels.
- Include definition, concept, explanation, comparison, and
  application-style questions where relevant.
- Avoid duplicate questions.
- Questions should sound natural like a real viva examiner.
- Do NOT provide answers.
- Return ONLY a valid JSON array.

Each item must have exactly this format:

{{
    "question": "Your viva question",
    "difficulty": "Easy"
}}

Difficulty must be one of:
"Easy", "Medium", or "Hard".

STUDY MATERIAL:
{source_text}

JSON:
"""

    max_retries = 3

    for attempt in range(max_retries):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_tokens=5000
            )

            raw_output = response.choices[0].message.content

            cleaned_output = _clean_json_output(raw_output)

            viva_data = json.loads(cleaned_output)

            if not isinstance(viva_data, list):
                raise ValueError(
                    "Invalid viva question format."
                )

            valid_questions = []

            for item in viva_data:

                if not isinstance(item, dict):
                    continue

                question = item.get(
                    "question",
                    ""
                ).strip()

                difficulty = item.get(
                    "difficulty",
                    ""
                ).strip()

                if not question:
                    continue

                if difficulty not in [
                    "Easy",
                    "Medium",
                    "Hard"
                ]:
                    difficulty = "Medium"

                valid_questions.append(
                    {
                        "question": question,
                        "difficulty": difficulty
                    }
                )

            if len(valid_questions) < num_questions:
                raise ValueError(
                    f"Only {len(valid_questions)} valid "
                    f"questions were generated."
                )

            return valid_questions[:num_questions]

        except RateLimitError:

            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                raise ValueError(
                    "Groq rate limit reached. "
                    "Please wait a moment and try again."
                )

        except json.JSONDecodeError:

            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise ValueError(
                    "Viva generator returned an invalid "
                    "JSON format. Please try again."
                )

        except ValueError:

            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

        except Exception as e:

            raise ValueError(
                f"Viva question generation failed: {str(e)}"
            )

    return []


def evaluate_viva_answer(
    question,
    student_answer,
    source_text
):
    """
    Evaluate a student's viva answer using the study material.
    """

    if (
        not question
        or not student_answer
        or not student_answer.strip()
    ):
        return {
            "evaluation": "No answer provided.",
            "score": 0,
            "feedback": (
                "Please provide an answer "
                "before evaluation."
            ),
            "suggested_answer": ""
        }

    # Limit source text before sending it to Groq
    source_text = _limit_source_text(source_text)

    client = get_groq_client()

    prompt = f"""
You are evaluating a B.Tech CSE student's viva answer.

Evaluate the student's answer ONLY using the provided study material.

QUESTION:
{question}

STUDENT ANSWER:
{student_answer}

STUDY MATERIAL:
{source_text}

Evaluate the answer fairly.

Scoring:
- 0 = Incorrect or irrelevant
- 1 = Partially correct
- 2 = Mostly correct
- 3 = Correct and sufficiently explained

Rules:
- Do not penalize the student for using different wording.
- Focus on whether the important concept is correct.
- Do not introduce facts that are not supported by the study material.
- Give concise and constructive feedback.
- Provide a better answer based only on the study material.

Return ONLY valid JSON in exactly this format:

{{
    "evaluation": "Correct / Mostly Correct / Partially Correct / Incorrect",
    "score": 0,
    "feedback": "Brief explanation of what was correct or what needs improvement.",
    "suggested_answer": "A clear improved answer based only on the study material."
}}

JSON:
"""

    max_retries = 3

    for attempt in range(max_retries):

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1500
            )

            raw_output = response.choices[0].message.content

            cleaned_output = _clean_json_output(
                raw_output
            )

            result = json.loads(cleaned_output)

            if not isinstance(result, dict):
                raise ValueError(
                    "Invalid evaluation format."
                )

            required_fields = [
                "evaluation",
                "score",
                "feedback",
                "suggested_answer"
            ]

            for field in required_fields:

                if field not in result:
                    raise ValueError(
                        f"Missing evaluation field: {field}"
                    )

            try:
                result["score"] = int(
                    result["score"]
                )
            except (TypeError, ValueError):
                result["score"] = 0

            result["score"] = max(
                0,
                min(3, result["score"])
            )

            return result

        except RateLimitError:

            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                raise ValueError(
                    "Groq rate limit reached. "
                    "Please wait a moment and try again."
                )

        except json.JSONDecodeError:

            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise ValueError(
                    "Viva evaluation returned an invalid "
                    "JSON format. Please try again."
                )

        except ValueError:

            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

        except Exception as e:

            raise ValueError(
                f"Viva answer evaluation failed: {str(e)}"
            )

    return {
        "evaluation": "Unable to evaluate",
        "score": 0,
        "feedback": "Please try again.",
        "suggested_answer": ""
    }