import os
import re

import anthropic


def classify_task(instruction: str) -> str:
    """
    Classify the task.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    prompt = f"""Classify the following task: {instruction}.

You can only pick one of the following classifications:
- FIGURE_EXTRACTION: This task is about extracting information from a figure. Any time the task is about extracting information from a file/figure, you should pick this classification.
- QA_ANSWERING: This task is about answering a question based on a file. Any time the task is about answering a question based on a file, you should pick this classification.
- OTHER: When you cannot classify the task into one of the above classifications, you should pick this classification.

ONLY return the classification in the following format:
{{
    "task_classification": "task_classification",
}}"""
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )

    # Clean up the response to extract just the JSON
    text = response.content[0].text

    # Remove triple quotes
    text = text.replace('"""', "")

    # Remove markdown code blocks
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Strip whitespace
    text = text.strip()

    return text
