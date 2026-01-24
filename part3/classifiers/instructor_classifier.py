
from part3.classifiers.base import BaseClassifier
import numpy as np

class InstructorClassifier(BaseClassifier):
    """
    Classifier using the Instructor framework.

    NOTE: This classifier is currently a placeholder. The 'instructor' library
    and its dependency 'jiter' have a conflict with Python 3.13, which this
    project uses. The 'jiter' package fails to build from source because its
    Rust-based components (via PyO3) do not yet support the Python 3.13 C-API.

    Once the dependencies are updated to support Python 3.13, the implementation
    for this classifier can be completed.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.is_compatible = False
        print("="*80)
        print("WARNING: The Instructor classifier is not available due to an incompatibility")
        print("with the project's Python 3.13 environment. Skipping.")
        print("="*80)


    def predict(self, X_test):
        """
        Returns an array of 'not_implemented' messages, as the classifier
        cannot be run.
        """
        if self.dry_run:
            return self._predict_dry_run(X_test)
        
        return np.full(len(X_test), "not_implemented")

