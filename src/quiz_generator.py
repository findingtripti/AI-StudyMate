import json
import re
import time

from groq import RateLimitError

from src.rag_pipeline import get_groq_client


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "openai/gpt-oss-20b"

DEFAULT_QUESTION_COUNT = 20

MIN_QUESTION_COUNT = 20
MAX_QUESTION_COUNT = 50


# ============================================================
# CLEAN JSON RESPONSE
# ============================================================

def _clean_json_output(raw_output):
    """
    Clean common Markdown/code-fence formatting
    from the LLM response before JSON parsing.
    """

    if not raw_output:
        return ""

    text = raw_output.strip()

    # Remove ```json ... ``` or ``` ... ```
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


# ============================================================
# VALIDATE ONE QUESTION
# ============================================================

def _validate_question(question):
    """
    Validate the structure of one generated MCQ.
    """

    if not isinstance(question, dict):
        return False

    question_text = question.get("question")
    options = question.get("options")
    correct_answer = question.get("correct_answer")
    explanation = question.get("explanation")

    # Question must exist
    if not isinstance(question_text, str):
        return False

    if len(question_text.strip()) < 10:
        return False

    # Exactly 4 options
    if not isinstance(options, list):
        return False

    if len(options) != 4:
        return False

    # Every option must be text
    for option in options:
        if not isinstance(option, str):
            return False

        if not option.strip():
            return False

    # Correct answer must be an index from 0 to 3
    if not isinstance(correct_answer, int):
        return False

    if correct_answer < 0 or correct_answer > 3:
        return False

    # Explanation must exist
    if not isinstance(explanation, str):
        return False

    if len(explanation.strip()) < 5:
        return False

    return True


# ============================================================
# VALIDATE QUIZ
# ============================================================

def _validate_quiz(quiz_data, expected_count):
    """
    Validate the complete generated quiz.
    """

    if not isinstance(quiz_data, list):
        return False

    if len(quiz_data) == 0:
        return False

    valid_questions = []

    for question in quiz_data:

        if _validate_question(question):
            valid_questions.append(question)

    if len(valid_questions) < expected_count:
        return False

    return True


# ============================================================
# NORMALIZE QUESTION COUNT
# ============================================================

def _normalize_question_count(num_questions):
    """
    Keep the requested question count within
    the supported range.
    """

    try:
        num_questions = int(num_questions)

    except (TypeError, ValueError):
        return DEFAULT_QUESTION_COUNT

    if num_questions < MIN_QUESTION_COUNT:
        return MIN_QUESTION_COUNT

    if num_questions > MAX_QUESTION_COUNT:
        return MAX_QUESTION_COUNT

    return num_questions


# ============================================================
# GENERATE QUIZ
# ============================================================

def generate_quiz(
    source_text,
    num_questions=DEFAULT_QUESTION_COUNT
):
    """
    Generate MCQs from the supplied study material.

    Supports 20 to 50 questions.

    Returns:
        list of dictionaries
    """

    if not source_text or not source_text.strip():
        return []

    num_questions = _normalize_question_count(
        num_questions
    )

    client = get_groq_client()

    prompt = f"""
You are an expert educational quiz generator
for AI StudyMate.

Create exactly {num_questions} multiple-choice questions
from the study material provided below.

IMPORTANT RULES:

1. Use ONLY information present in the study material.
2. Do not use outside knowledge.
3. Each question must have exactly 4 options.
4. Only ONE option must be correct.
5. Make the questions useful for a B.Tech CSE student.
6. Mix question types where possible:
   - definition
   - concept understanding
   - how/why
   - comparison
   - application
7. Avoid duplicate or nearly identical questions.
8. Do not create questions from table-of-contents text.
9. Do not mention that you are an AI.
10. Keep the language clear and student-friendly.
11. Cover different parts of the provided material.
12. Do not repeat the same concept unnecessarily.

OUTPUT FORMAT:

Return ONLY valid JSON.

Do not use Markdown.
Do not use ```json.
Do not add any explanation outside the JSON.

The JSON must be an array in exactly this structure:

[
  {{
    "question": "Question text?",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "correct_answer": 0,
    "explanation": "Why this option is correct."
  }}
]

IMPORTANT:

- correct_answer must be the zero-based option index.
- 0 = first option
- 1 = second option
- 2 = third option
- 3 = fourth option

STUDY MATERIAL:

{source_text}

GENERATE EXACTLY {num_questions} QUESTIONS NOW.
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

                # Increased because 20-50 MCQs
                # need more output space.
                max_tokens=12000
            )

            raw_output = (
                response.choices[0]
                .message.content
            )

            cleaned_output = _clean_json_output(
                raw_output
            )

            quiz_data = json.loads(
                cleaned_output
            )

            if _validate_quiz(
                quiz_data,
                num_questions
            ):

                valid_questions = [
                    question
                    for question in quiz_data
                    if _validate_question(question)
                ]

                return valid_questions[:num_questions]

            raise ValueError(
                "The generated quiz did not contain "
                f"{num_questions} valid questions."
            )

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
                    "The quiz generator returned "
                    "an invalid JSON format. "
                    "Please try again."
                )

        except ValueError:

            if attempt < max_retries - 1:

                time.sleep(2)

            else:

                raise

        except Exception as e:

            raise ValueError(
                f"Quiz generation failed: {str(e)}"
            )

    return []