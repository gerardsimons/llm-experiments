from pprint import pprint

from sklearn.metrics import classification_report
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier

from part3.data_loader import load_data

X_train, y_train, X_test, y_test = load_data(100, 10)

# Initialise dynamicfewshotclassifier using ollama vectorizer
# Default vectorizer uses nomic-embed-text
clf = DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=3)

print("Fitting data")
clf.fit(X_train, y_train)

print("Predicting")
y_pred = clf.predict(X_test)

pprint(classification_report(y_test, y_pred))