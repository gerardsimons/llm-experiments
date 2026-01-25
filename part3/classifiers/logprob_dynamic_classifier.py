import numpy as np
import math

from datasets import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import entropy

# Assuming get_logprobs is available and works as in your repo
from part2.logprobs import get_logprobs, get_logprobs_cached


class LogprobDynamicFewShotClassifier(BaseEstimator, ClassifierMixin):
    """
    A dynamic few-shot classifier that selects examples based on model uncertainty
    derived from log probabilities.
    """

    def __init__(self, model="ollama/llama3:8b", n_examples=3, selection_strategy="entropy", verbose=False, **kwargs):
        """
        Args:
            model (str): The model to use (e.g., 'ollama/llama3:8b').
            n_examples (int): The number of examples to select for the few-shot prompt.
            selection_strategy (str): The strategy to use for selecting examples.
                                      Options: 'entropy', 'margin', 'least_confidence'.
        """
        self.model = model
        self.n_examples = n_examples
        self.selection_strategy = selection_strategy
        if "/" in model:
            self.provider, self.model_id = model.split('/')
        else:
            self.provider = "ollama"
            self.model_id = model
        self.verbose = verbose

    def fit(self, X, y):
        """
        Selects the most informative examples from the training data based on the
        chosen selection strategy.
        """
        self.classes_ = np.unique(y)

        scores = []
        if self.verbose:
            print(f"Fitting with {self.selection_strategy} strategy...")

        show_pbar = True

        for i, text in tqdm(enumerate(X), total=len(X), disable=not show_pbar):
            # For each training example, get the logprobs for all possible labels
            # We'll use a simple prompt for this.
            prompt = f"Text: '{text.replace('\n', ' ')}'\\nWhat is the label? Answer with one of: {', '.join(self.classes_)}"

            top_logprobs = min(len(self.classes_) * 2, 10)  # Get enough to find all labels, but cannot exceed 10 on ollama3
            try:
                logprobs_result = get_logprobs_cached(
                    provider=self.provider,
                    model_id=self.model_id,
                    prompt=prompt,
                    top_logprobs=top_logprobs,
                    invert_log=False
                )
            except Exception as e:
                print(f"Error processing text: {text}")
                print(f"Prompt: {prompt}")
                raise e

            # Extract probabilities for each class
            class_logprobs = np.array([logprobs_result.logprobs.get(label, -20) for label in self.classes_])
            class_probs = np.exp(class_logprobs)
            # Normalize to get a probability distribution
            class_probs_sum = class_probs.sum()
            if class_probs_sum > 0:
                class_probs = class_probs / class_probs_sum

            # Calculate score based on strategy
            if self.selection_strategy == "entropy":
                score = entropy(class_probs)
            elif self.selection_strategy == "margin":
                sorted_probs = np.sort(class_probs)[::-1]
                score = 1 - (sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1
            elif self.selection_strategy == "least_confidence":
                score = 1 - np.max(class_probs)
            else:
                raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")

            scores.append(score)

        # Select top n_examples with the highest scores (most uncertainty)
        top_indices = np.argsort(scores)[-self.n_examples:]
        self.few_shot_examples_ = [(X.iloc[i], y.iloc[i]) for i in top_indices]

        return self


    def predict(self, X):
        """
        Predicts labels for new data using the selected few-shot examples.
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=str, ensure_2d=False)

        predictions = []

        example_prompt_part = "\n".join(
            [f"Text: '{text}'\nLabel: {label}" for text, label in self.few_shot_examples_])

        for text in X:
            prompt = (
                f"You are a text classifier. Here are some examples:\n"
                f"{example_prompt_part}\n\n"
                f"Now, classify the following text. Provide only the label.\n"
                f"Text: '{text}'\n"
                f"Label:"
            )

            # We don't need logprobs here, just the generated text
            response = get_logprobs_cached(
                provider=self.provider,
                model_id=self.model_id,
                prompt=prompt,
                temperature=0,
                top_logprobs=1
            )

            predictions.append(response.response_text.strip())

        return np.array(predictions)
