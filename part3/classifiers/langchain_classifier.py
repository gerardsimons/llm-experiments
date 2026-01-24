
from .base import BaseClassifier
import numpy as np
from rich.progress import track
import diskcache
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from part3.prompts import LANGCHAIN_PROMPT_TEMPLATE

# Define a cache directory and initialize cache
CACHE_DIR = "part3/.llm_cache/langchain"
cache = diskcache.Cache(CACHE_DIR)


class LangChainClassifier(BaseClassifier):
    """
    Classifier using the LangChain framework.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.llm = self._get_llm()
        self.prompt_template = PromptTemplate.from_template(LANGCHAIN_PROMPT_TEMPLATE)
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _get_llm(self):
        """Dynamically loads the LLM based on the model name."""
        # This is a simple factory. A more robust solution could be added.
        if self.model.startswith('openai/'):
            from langchain_openai import ChatOpenAI
            model_name = self.model.split('/')[1]
            return ChatOpenAI(model=model_name, temperature=0)
        elif self.model.startswith('ollama/'):
            from langchain_ollama import ChatOllama
            model_name = self.model.split('/')[1]
            return ChatOllama(model=model_name, temperature=0)
        else:
            raise ValueError(
                f"Unsupported model provider for LangChain. Model name should start with 'openai/' or 'ollama/'. Got: {self.model}"
            )

    def fit(self, X_train, y_train):
        """Stores the class labels."""
        super().fit(X_train, y_train)
        print("LangChain classifier fitted (class labels stored).")

    def predict(self, X_test):
        """Predicts labels for the test data using the LCEL chain."""
        if self.dry_run:
            return self._predict_dry_run(X_test)

        if self.show_prompt and len(X_test) > 0:
            self._prompt_to_display = self.prompt_template.format(
                labels=', '.join(self.classes_),
                text=X_test.iloc[0]
            )

        print("Predicting with LangChain...")

        inputs = [{"text": x, "labels": ', '.join(self.classes_)} for x in X_test]
        cache_keys = [f"{self.model}-{str(input_dict)}" for input_dict in inputs]

        # Try to get all predictions from cache
        try:
            cached_predictions = [cache.get(key) for key in cache_keys]
        except Exception as e:
            print(f"Could not read from cache: {e}")
            cached_predictions = [None] * len(cache_keys)

        # Identify what's missing
        missing_indices = [i for i, p in enumerate(cached_predictions) if p is None]
        predictions = cached_predictions

        if missing_indices:
            print(f"Cache miss for {len(missing_indices)} out of {len(X_test)} items. Calling LLM...")
            missing_inputs = [inputs[i] for i in missing_indices]

            try:
                # Run batch for missing items
                new_predictions = self.chain.batch(missing_inputs, {"max_concurrency": 5})

                # Store new predictions in cache and in the full predictions list
                for i, new_pred in zip(missing_indices, new_predictions):
                    try:
                        cache.set(cache_keys[i], new_pred)
                    except Exception as e:
                        print(f"Could not write to cache: {e}")
                    predictions[i] = new_pred
            except Exception as e:
                print(f"An error occurred during LangChain prediction: {e}")
                print("Falling back to sequential prediction for missing items.")
                # Fallback to sequential invocation if batch fails
                for i in track(missing_indices, description="Predicting sequentially..."):
                    try:
                        result = self.chain.invoke(inputs[i])
                        try:
                            cache.set(cache_keys[i], result)
                        except Exception as e:
                            print(f"Could not write to cache: {e}")
                        predictions[i] = result
                    except Exception as invoke_e:
                        print(f"Error on sample: {inputs[i]['text']}. Error: {invoke_e}. Appending 'error' as prediction.")
                        predictions[i] = "error"
        else:
            print("All items found in cache.")

        # Clean up all predictions
        cleaned_predictions = [p.strip().split('\\n')[0] for p in predictions]
        return np.array(cleaned_predictions)
