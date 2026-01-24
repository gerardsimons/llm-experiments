
from abc import ABC, abstractmethod
import numpy as np

class BaseClassifier(ABC):
    """Abstract base class for all framework classifiers."""

    def __init__(self, model: str, dry_run: bool = False, show_prompt: bool = False):
        self.model = model
        self.dry_run = dry_run
        self.show_prompt = show_prompt
        self.classes_ = np.array([])
        self._prompt_to_display = None

    @abstractmethod
    def fit(self, X_train, y_train):
        """
        Fit the classifier to the training data.
        In many LLM cases, this might just involve storing class labels or examples.
        """
        self.classes_ = np.unique(y_train)
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict labels for the test data.
        """
        pass
    
    def _predict_dry_run(self, X_test):
        """
        Generates random predictions for a dry run.
        """
        print("Performing a dry run with random predictions...")
        if len(self.classes_) == 0:
            raise RuntimeError("Classifier must be fitted before prediction, even for a dry run, to know the possible labels.")
        return np.random.choice(self.classes_, size=len(X_test))

    def get_prompt_to_display(self):
        """Returns the prompt that was generated for the first item in the last predict call."""
        if self._prompt_to_display:
            return self._prompt_to_display
        else:
            return "No prompt was generated or stored. Make sure `show_prompt` is enabled and `predict` has been called."

