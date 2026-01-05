import requests
import json
from typing import List, Dict
from collections import Counter
import random

from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
BASE_PROMPT = """
You are classifying restaurant reviews by sentiment.

Labels:
- positive
- neutral
- negative

Instructions:
Return exactly one label from the list above.
Do not add explanations.

Review:
"{review}"
"""

PROMPT_VARIANTS = [
    """
Classify the sentiment of the restaurant review below.

Possible labels: positive, neutral, negative.

Return only the label.

Review:
"{review}"
""",
    """
Determine the sentiment category for the following restaurant review.

Choose one of: positive | neutral | negative.
Respond with the label only.

Text:
"{review}"
""",
    """
Task: sentiment classification.

Output exactly one of these labels:
positive, neutral, negative

Restaurant review:
"{review}"
""",
    """
Assign a sentiment label to the review below.

Valid labels:
- positive
- neutral
- negative

Only output the label.

Review:
"{review}"
""",
]

def run_stability_test(
    review: str,
    model: str,
    n_runs: int = 20,
    temperature: float = 0.0,
) -> Dict[str, int]:

    outputs = []

    for _ in tqdm(range(n_runs)):
        prompt_template = random.choice(PROMPT_VARIANTS)
        prompt = prompt_template.format(review=review)

        label = call_ollama(
            model=model,
            prompt=prompt,
            temperature=temperature,
        )

        # post process label
        label = normalize_label(label)

        outputs.append(label)

    return Counter(outputs)

def normalize_label(output: str) -> str:
    output = output.lower().strip()
    for label in ["positive", "neutral", "negative"]:
        print(output)
        if label == output:
            return label
    return "invalid"


def call_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()


if __name__ == "__main__":
    review = (
        "The food was decent but overpriced. Interior was great, but service was slow, "
        "and the waiter seemed somewhat distracted. Probably won't come back."
    )

    # Available models:
    # ['llama3:8b',
    #  'nous-hermes:latest',
    #  'dolphin-mistral:latest',
    #  'deepseek-r1:14b']
    models = ['llama3:8b', 'dolphin-mistral:latest']
    for model in models:
        print("Model:", model)
        results = run_stability_test(
            review=review,
            model=model,
            n_runs=10,
            temperature=0,
        )

        print("Stability results:")
        for label, count in results.items():
            print(f"{label}: {count}")

