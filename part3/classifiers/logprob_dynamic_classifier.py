import numpy as np
import pandas as pd
from skllm.prompts.builders import build_few_shot_prompt_slc
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import json

from part2.logprobs import get_logprobs_cached
from skllm.prompts.templates import FEW_SHOT_CLF_PROMPT_TEMPLATE


class LogprobDynamicFewShotClassifier(BaseEstimator, ClassifierMixin):
    """
    Few-shot classifier using logprob-based selection strategies.
    """

    def __init__(
        self,
        model="ollama/llama3:8b",
        n_examples=3,
        strategy="profile_conditional",
        distance="l2",
        verbose=False,
        prompt_template=FEW_SHOT_CLF_PROMPT_TEMPLATE,
        self_sample=False,
    ):
        self.model = model
        self.n_examples = n_examples
        self.strategy = strategy
        self.distance = distance
        self.verbose = verbose
        self.prompt_template = prompt_template
        self.self_sample = self_sample

        if "/" in model:
            self.provider, self.model_id = model.split("/")
        else:
            self.provider = "ollama"
            self.model_id = model

    # -------------------------
    # Utilities
    # -------------------------

    def _label_prompt(self, text):
        labels = ", ".join(self.classes_)
        return (
            f"Text: '{text.replace(chr(10), ' ')}'\n"
            f"What is the label? Answer with one of: {labels}. Output only the label."
        )

    def _distance(self, a, b):
        if self.distance == "l2":
            return np.linalg.norm(a - b)
        if self.distance == "cosine":
            return 1 - np.dot(a, b) / (
                np.linalg.norm(a) * np.linalg.norm(b) + 1e-8
            )
        raise ValueError(f"Unknown distance: {self.distance}")

    def _logprobs(self, text):
        res = get_logprobs_cached(
            provider=self.provider,
            model_id=self.model_id,
            prompt=self._label_prompt(text),
            top_logprobs=min(len(self.classes_) * 2, 10),
            temperature=0,
        )
        return np.array(
            [res.logprobs.get(lbl, -20.0) for lbl in self.classes_]
        )

    # -------------------------
    # Fit
    # -------------------------

    def fit(self, X, y):
        X = check_array(X, dtype=str, ensure_2d=False)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        self.logprob_profiles_ = []
        self.entropies_ = []
        self.margins_ = []
        self.top2_ = []

        if self.verbose:
            print("Computing decision-space statistics...")

        for text in tqdm(X, disable=not self.verbose):
            logps = self._logprobs(text)

            probs = np.exp(logps - logps.max())
            probs /= probs.sum()

            order = np.argsort(logps)[::-1]
            margin = logps[order[0]] - logps[order[1]]

            self.logprob_profiles_.append(logps)
            self.entropies_.append(entropy(probs))
            self.margins_.append(margin)
            self.top2_.append(order[:2])

        self.logprob_profiles_ = np.vstack(self.logprob_profiles_)
        self.entropies_ = np.asarray(self.entropies_)
        self.margins_ = np.asarray(self.margins_)
        self.top2_ = np.asarray(self.top2_)

        # Global strategies preselect once
        if self.strategy == "entropy_global":
            idx = np.argsort(self.entropies_)[-self.n_examples :]
            self.global_examples_ = idx

        elif self.strategy == "margin_global":
            idx = np.argsort(self.margins_)[: self.n_examples]
            self.global_examples_ = idx

        return self

    # -------------------------
    # Selection
    # -------------------------

    def _select_examples(self, test_logps):
        if self.strategy in {"entropy_global", "margin_global"}:
            return self.global_examples_

        if self.strategy == "profile_conditional":
            distances = [
                self._distance(test_logps, lp)
                for lp in self.logprob_profiles_
            ]
            return np.argsort(distances)[: self.n_examples]

        if self.strategy == "confusion_conditional":
            order = np.argsort(test_logps)[::-1][:2]
            mask = np.all(self.top2_ == order, axis=1)
            idx = np.where(mask)[0]

            if len(idx) >= self.n_examples:
                return idx[: self.n_examples]

            # fallback: nearest in profile space
            distances = [
                self._distance(test_logps, lp)
                for lp in self.logprob_profiles_
            ]
            return np.argsort(distances)[: self.n_examples]

        raise ValueError(f"Unknown strategy: {self.strategy}")

    # -------------------------
    # Predict
    # -------------------------

    def predict(self, X):
        if self.n_examples:
            check_is_fitted(self)
        X = check_array(X, dtype=str, ensure_2d=False)

        preds = []

        for text in X:
            test_logps = self._logprobs(text)
            idx = self._select_examples(test_logps)

            examples = [(self.X_[i], self.y_[i]) for i in idx]
            training_data = "\n".join(
                [f"Text: ```{x}```\nLabel: {y}" for x, y in examples]
            )

            prompt = self.prompt.format(
                labels=str(list(self.classes_)),
                training_data=training_data,
                x=text,
            )

            out = get_logprobs_cached(
                provider=self.provider,
                model_id=self.model_id,
                prompt=prompt,
                temperature=0,
                top_logprobs=min(len(self.classes_) * 2, 10), # request enough logprobs to find all possible labels
            )

            if self.self_sample:
                # If self_sample is True, we choose the label with the highest logprob
                # This assumes that the labels are single tokens.
                # TODO: This will be problematic for multi-token outputs.
                if out.logprobs:
                    # logprobs is an OrderedDict, so the first key is the highest prob token
                    predicted_label = list(out.logprobs.keys())[0]
                    preds.append(predicted_label)
                else:
                    # Fallback if no logprobs are returned
                    preds.append(out.response_text.strip())
            else:
                # Original behavior: parse JSON from response_text
                try:
                    response_json = json.loads(out.response_text.strip())
                    preds.append(response_json["label"])
                except (json.JSONDecodeError, KeyError):
                    preds.append(out.response_text.strip())

        return np.asarray(preds)

if __name__ == '__main__':
    clf = LogprobDynamicFewShotClassifier(n_examples=0)
    df = pd.DataFrame({'text':["Buy here more", "Hey, how are you?"], 'labels': ["spam", "ham"]})
    X = df['text']
    y = df['labels']
    clf.fit(X, y)
    pred = clf.predict(X)
    print(pred)