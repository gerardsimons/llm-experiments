import numpy as np
import pandas as pd
from skllm.prompts.builders import build_few_shot_prompt_slc, build_zero_shot_prompt_slc
from tqdm import tqdm
from scipy.stats import entropy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
import json

from part2.logprobs import get_logprobs_cached
from part3.prompts import NEWS_FEW_SHOT_PROMPT, NEWS_ZERO_SHOT_PROMPT


class LogprobDynamicFewShotClassifier(BaseEstimator, ClassifierMixin):
    """
    Few-shot classifier using logprob-based selection strategies.
    """

    def __init__(
        self,
        prompt_templates,
        model="ollama/llama3:8b",
        n_examples=3,
        strategy="profile_conditional",
        distance="l2",
        verbose=False,
        self_sample=False,
        example_template="Text: ```{x}```\nLabel: {y}",
    ):
        self.model = model
        self.n_examples = n_examples
        self.strategy = strategy
        self.distance = distance
        self.verbose = verbose
        self.few_shot_prompt_template = prompt_templates['few']
        self.zero_shot_prompt_template = prompt_templates['zero']
        self.self_sample = self_sample
        self.example_template = example_template

        if "/" in model:
            self.provider, self.model_id = model.split("/")
        else:
            self.provider = "ollama"
            self.model_id = model

    # -------------------------
    # Utilities
    # -------------------------

    def _distance(self, a, b):
        if self.distance == "l2":
            return np.linalg.norm(a - b)
        if self.distance == "cosine":
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        raise ValueError(f"Unknown distance: {self.distance}")

    def _get_zero_shot_logprobs(self, text):
        """Always gets logprobs using a zero-shot prompt."""
        prompt = self._create_zero_shot_prompt(text)
        # print(">>>", prompt)
        res = get_logprobs_cached(
            provider=self.provider,
            model_id=self.model_id,
            prompt=prompt,
            top_logprobs=min(len(self.classes_) * 2, 10),
            temperature=0,
        )
        return np.array([res.logprobs.get(lbl, -20.0) for lbl in self.classes_])

    # -------------------------
    # Fit
    # -------------------------

    def fit(self, X, y):
        X = check_array(X, dtype=str, ensure_2d=False)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        if self.n_examples == 0:
            return self

        self.logprob_profiles_ = []
        self.entropies_ = []
        self.margins_ = []
        self.top2_ = []

        if self.strategy == "random":
            return self

        if self.verbose:
            print("Computing decision-space statistics...")

        for text in tqdm(
            X, disable=not self.verbose, desc="Fitting LogProbDynamic"
        ):
            logps = self._get_zero_shot_logprobs(text)

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

        return self

    # -------------------------
    # Selection
    # -------------------------

    def _select_examples(self, test_logps=None):
        selected_indices = []
        n_per_class = self.n_examples // len(self.classes_)
        remainder = self.n_examples % len(self.classes_)

        if self.strategy == "random":
            for i, label in enumerate(self.classes_):
                class_indices = np.where(self.y_ == label)[0]
                num_to_select = n_per_class + (1 if i < remainder else 0)
                random_indices = np.random.choice(
                    class_indices, size=num_to_select, replace=False
                )
                selected_indices.extend(random_indices)
            return selected_indices

        if "global" in self.strategy:
            if self.strategy == "entropy_global":
                scores = self.entropies_
                ascending = True
            elif self.strategy == "margin_global":
                scores = self.margins_
                ascending = False
            else:
                raise ValueError(f"Unknown global strategy: {self.strategy}")

            for i, label in enumerate(self.classes_):
                class_indices = np.where(self.y_ == label)[0]
                class_scores = scores[class_indices]

                num_to_select = n_per_class + (1 if i < remainder else 0)

                if ascending:
                    top_class_indices = class_indices[
                        np.argsort(class_scores)[-num_to_select:]
                    ]
                else:
                    top_class_indices = class_indices[
                        np.argsort(class_scores)[:num_to_select]
                    ]

                selected_indices.extend(top_class_indices)
            return selected_indices

        elif "conditional" in self.strategy:
            if self.strategy == "profile_conditional":
                distances = np.array(
                    [self._distance(test_logps, lp) for lp in self.logprob_profiles_]
                )
                scores = distances
                ascending = True
            elif self.strategy == "confusion_conditional":
                order = np.argsort(test_logps)[::-1][:2]
                mask = np.all(self.top2_ == order, axis=1)

                scores = self.margins_
                ascending = True

                scores[~mask] = np.inf if ascending else -np.inf

            else:
                raise ValueError(f"Unknown conditional strategy: {self.strategy}")

            for i, label in enumerate(self.classes_):
                class_indices = np.where(self.y_ == label)[0]
                class_scores = scores[class_indices]

                num_to_select = n_per_class + (1 if i < remainder else 0)

                if ascending:
                    top_class_indices = class_indices[
                        np.argsort(class_scores)[:num_to_select]
                    ]
                else:
                    top_class_indices = class_indices[
                        np.argsort(class_scores)[-num_to_select:]
                    ]

                selected_indices.extend(top_class_indices)
            return selected_indices

        raise ValueError(f"Unknown strategy: {self.strategy}")

    # -------------------------
    # Prompting
    # -------------------------

    def _create_zero_shot_prompt(self, text):
        return build_zero_shot_prompt_slc(
            x=text,
            labels=str(list(self.classes_)),
            template=self.zero_shot_prompt_template,
        )

    def _create_few_shot_prompt(self, text):
        test_logps = (
            self._get_zero_shot_logprobs(text)
            if "conditional" in self.strategy
            else None
        )
        idx = self._select_examples(test_logps)

        examples = [(self.X_[i], self.y_[i]) for i in idx]

        examples_by_class = {label: [] for label in self.classes_}
        for x, y in examples:
            examples_by_class[y].append((x, y))

        interleaved_examples = []
        max_len = (
            max(len(v) for v in examples_by_class.values())
            if examples_by_class
            else 0
        )
        for i in range(max_len):
            for label in self.classes_:
                if label in examples_by_class and i < len(examples_by_class[label]):
                    interleaved_examples.append(examples_by_class[label][i])

        training_data = "\n".join(
            [self.example_template.format(x=x, y=y) for x, y in interleaved_examples]
        )
        return build_few_shot_prompt_slc(
            x=text,
            labels=str(list(self.classes_)),
            training_data=training_data,
            template=self.few_shot_prompt_template,
        )

    def create_prompt(self, text) -> str:
        if self.n_examples > 0:
            return self._create_few_shot_prompt(text)
        return self._create_zero_shot_prompt(text)

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype=str, ensure_2d=False)

        preds = []

        for text in tqdm(X, desc="LogProb predict"):
            prompt = self.create_prompt(text)

            out = get_logprobs_cached(
                provider=self.provider,
                model_id=self.model_id,
                prompt=prompt,
                temperature=0,
                top_logprobs=min(len(self.classes_) * 2, 10),
            )

            if self.self_sample:
                if out.logprobs:
                    predicted_label = list(out.logprobs.keys())[0]
                    preds.append(predicted_label)
                else:
                    preds.append(out.response_text.strip())
            else:
                try:
                    response_json = json.loads(out.response_text.strip())
                    preds.append(response_json["label"])
                except (json.JSONDecodeError, KeyError):
                    preds.append(out.response_text.strip())

        return np.asarray(preds)



def try_zeroshot():
    print("--- Running Zero-Shot Example ---")
    spam_prompt_template = """You are classifying SMS messages.

    Question: Is the following SMS message spam? The possible labels are: {labels}

    Answer with exactly one token: Yes or No.

    Message:
    `{x}`

    Answer:
    """

    clf = LogprobDynamicFewShotClassifier(
        n_examples=0,
        zero_shot_prompt_template=spam_prompt_template,
        self_sample=True,
    )
    df = pd.DataFrame(
        {"text": ["Buy here more", "Hey, how are you?"], "labels": ["Yes", "No"]}
    )
    X = df["text"]
    y = df["labels"]

    clf.fit(X, y)
    pred = clf.predict(X)
    print("Predictions:", pred)
    print("-" * 20)


def try_fewshot():
    print("\n--- Running Few-Shot Example ---")
    clf = LogprobDynamicFewShotClassifier(
        n_examples=2,
        strategy="entropy_global",
        self_sample=False,
    )
    data = {
        "text": [
            "Get a free iPhone now!",
            "Meeting at 5pm tomorrow.",
            "URGENT: Your account has been compromised. Click here.",
            "Can you pick up milk on your way home?",
            "Exclusive offer just for you!",
            "Happy Birthday!",
        ],
        "labels": ["spam", "ham", "spam", "ham", "spam", "ham"],
    }
    df = pd.DataFrame(data)
    X = df["text"]
    y = df["labels"]

    clf.fit(X, y)

    sample = X.iloc[0]
    print("Sample Prompt:", clf.create_prompt(sample))

    new_text = ["Hi mom, how are you doing?"]
    pred = clf.predict(np.array(new_text))
    print(f"Prediction for '{new_text[0]}':", pred)
    print("-" * 20)


def try_random():
    print("\n--- Running Random Stratified Example ---")
    clf = LogprobDynamicFewShotClassifier(
        n_examples=2,
        strategy="random",
        self_sample=False,
    )
    data = {
        "text": [
            "Get a free iPhone now!",
            "Meeting at 5pm tomorrow.",
            "URGENT: Your account has been compromised. Click here.",
            "Can you pick up milk on your way home?",
            "Exclusive offer just for you!",
            "Happy Birthday!",
            "You won a lottery!",
            "Let's catch up later."
        ],
        "labels": ["spam", "ham", "spam", "ham", "spam", "ham", "spam", "ham"],
    }
    df = pd.DataFrame(data)
    X = df["text"]
    y = df["labels"]

    clf.fit(X, y)

    sample = X.iloc[0]
    print("Sample Prompt:", clf.create_prompt(sample))

    new_text = ["Hi mom, how are you doing?"]
    pred = clf.predict(np.array(new_text))
    print(f"Prediction for '{new_text[0]}':", pred)
    print("-" * 20)


if __name__ == "__main__":
    try_zeroshot()
    try_fewshot()
    try_random()