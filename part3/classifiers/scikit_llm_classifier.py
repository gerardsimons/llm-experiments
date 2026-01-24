
from .base import BaseClassifier
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
from skllm.models.gpt.classification.few_shot import FewShotGPTClassifier, DynamicFewShotGPTClassifier
from part3.prompts import SKLLM_ZERO_SHOT_PROMPT, SKLLM_FEW_SHOT_PROMPT
import numpy as np
from rich.progress import track
import diskcache

# Define a cache directory and initialize cache
CACHE_DIR = "part3/.llm_cache/scikit-llm"
cache = diskcache.Cache(CACHE_DIR)


# Note: To fulfill the requirement of showing and modifying prompts, we have to
# create a custom class and override the private `_get_prompt` method. This is
# because scikit-llm does not expose prompt templating in its public API.

class CustomZeroShotOllamaClassifier(ZeroShotOllamaClassifier):
    def __init__(self, model="llama3:8b", custom_prompt=None):
        super().__init__(model=model)
        self.custom_prompt = custom_prompt or SKLLM_ZERO_SHOT_PROMPT

    def _get_prompt(self, x: str) -> str:
        """Overrides the default prompt builder to use our custom template."""
        labels = self.classes_.tolist()
        return self.custom_prompt.format(labels=labels, text=x)

class ScikitLLMClassifier(BaseClassifier):
    """
    Classifier for scikit-llm and scikit-ollama frameworks.
    """

    def __init__(self, model: str, mode: str = 'zero-shot', **kwargs):
        super().__init__(model, **kwargs)
        self.mode = mode
        self.classifier = self._get_classifier()

    def _get_classifier(self):
        """Initializes the correct scikit-llm classifier based on the mode."""
        if self.mode == 'zero-shot':
            # We assume ollama models for zero-shot as per the notebook
            return CustomZeroShotOllamaClassifier(model=self.model)
        elif self.mode == 'few-shot':
            print("Warning: scikit-llm's FewShotGPTClassifier is designed for OpenAI models.")
            # TODO: Add custom prompt support for few-shot if possible
            return FewShotGPTClassifier(model=self.model)
        elif self.mode == 'dynamic-shot':
            print("Warning: scikit-llm's DynamicFewShotGPTClassifier is designed for OpenAI models.")
             # TODO: Add custom prompt support for dynamic-shot if possible
            return DynamicFewShotGPTClassifier(model=self.model)
        else:
            raise ValueError(f"Unknown mode for ScikitLLMClassifier: {self.mode}")

    def fit(self, X_train, y_train):
        super().fit(X_train, y_train)
        if self.mode != 'zero-shot':
            print(f"Fitting scikit-llm {self.mode} classifier...")
            self.classifier.fit(X_train, y_train)
        else:
            # Zero-shot doesn't require fitting, but we fit the base class
            self.classifier.classes_ = self.classes_
            print("scikit-llm zero-shot classifier does not require fitting.")

    def predict(self, X_test):
        if self.dry_run:
            return self._predict_dry_run(X_test)
        
        # Generate prompt for the first item if requested
        if self.show_prompt and len(X_test) > 0:
            try:
                # We use our overridden method for zero-shot
                self._prompt_to_display = self.classifier._get_prompt(X_test.iloc[0])
            except Exception as e:
                self._prompt_to_display = f"Could not generate prompt: {e}"

        print(f"Predicting with scikit-llm {self.mode} classifier...")

        if self.mode == 'zero-shot':
            predictions = []
            for x in track(X_test, description="Predicting with scikit-llm..."):
                prompt = self.classifier._get_prompt(x)
                cache_key = f"{self.model}-{self.mode}-{prompt}"
                
                try:
                    cached_result = cache.get(cache_key)
                    if cached_result is not None:
                        predictions.append(cached_result)
                        continue
                except Exception as e:
                    print(f"Could not read from cache: {e}")

                try:
                    result = self.classifier.predict([x])[0]
                    try:
                        cache.set(cache_key, result)
                    except Exception as e:
                        print(f"Could not write to cache: {e}")
                    predictions.append(result)
                except Exception as e:
                    print(f"Error on sample: {x}. Error: {e}. Appending 'error' as prediction.")
                    predictions.append("error")
            return np.array(predictions)
        else:
            # For few-shot and dynamic-shot, we don't have an easy way to get the full prompt
            # so we can't reliably cache. We call predict on the whole set.
            print("Caching not implemented for few-shot/dynamic-shot modes in scikit-llm.")
            return self.classifier.predict(X_test)

