
from part3.classifiers.base import BaseClassifier
from part3.prompts import LLAMAINDEX_PROMPT_TEMPLATE
import numpy as np
from rich.progress import track
import diskcache
from llama_index.program.core import PydanticProgram
from pydantic import BaseModel, Field

# Define a cache directory and initialize cache
CACHE_DIR = "part3/.llm_cache/llamaindex"
cache = diskcache.Cache(CACHE_DIR)


class LanguagePrediction(BaseModel):
    """A Pydantic model for the language prediction."""
    label: str = Field(..., description="The predicted two-letter language code (e.g., 'en', 'fr').")

class LlamaIndexClassifier(BaseClassifier):
    """
    Classifier using the LlamaIndex framework.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.llm = self._get_llm()

    def _get_llm(self):
        """Dynamically loads the LLM based on the model name."""
        if self.model.startswith('openai/'):
            from llama_index.llms.openai import OpenAI
            model_name = self.model.split('/')[1]
            return OpenAI(model=model_name, temperature=0)
        elif self.model.startswith('ollama/'):
            from llama_index.llms.ollama import Ollama
            model_name = self.model.split('/')[1]
            return Ollama(model=model_name, temperature=0)
        else:
            raise ValueError(
                f"Unsupported model provider for LlamaIndex. Model name should start with 'openai/' or 'ollama/'. Got: {self.model}"
            )
            
    def fit(self, X_train, y_train):
        """Stores the class labels."""
        super().fit(X_train, y_train)
        print("LlamaIndex classifier fitted (class labels stored).")
        
        # Define the program with the dynamic labels from the data
        self.program = PydanticProgram(
            output_cls=LanguagePrediction,
            prompt_template_str=LLAMAINDEX_PROMPT_TEMPLATE.format(labels=', '.join(self.classes_), text="{text}"),
            llm=self.llm,
            verbose=self.show_prompt  # Show prompt generation in verbose mode
        )

    def predict(self, X_test):
        """Predicts labels for the test data using the PydanticProgram."""
        if self.dry_run:
            return self._predict_dry_run(X_test)
            
        if not hasattr(self, 'program'):
            raise RuntimeError("The classifier must be fitted before prediction to initialize the LlamaIndex program.")

        if self.show_prompt and len(X_test) > 0:
            # LlamaIndex's program verbosity handles showing the prompt, 
            # but we can construct one for display consistency.
             self._prompt_to_display = LLAMAINDEX_PROMPT_TEMPLATE.format(
                labels=', '.join(self.classes_),
                text=X_test.iloc[0]
            )

        print("Predicting with LlamaIndex...")
        predictions = []
        for x in track(X_test, description="Predicting..."):
            cache_key = f"{self.model}-{self.program.prompt.template}-{x}"
            try:
                cached_result = cache.get(cache_key)
                if cached_result is not None:
                    predictions.append(cached_result)
                    continue
            except Exception as e:
                print(f"Could not read from cache: {e}")

            try:
                result = self.program(text=x)
                prediction = result.label
                try:
                    cache.set(cache_key, prediction)
                except Exception as e:
                    print(f"Could not write to cache: {e}")
                predictions.append(prediction)
            except Exception as e:
                print(f"Error on sample: {x}. Error: {e}. Appending 'error' as prediction.")
                predictions.append("error")
        
        return np.array(predictions)
