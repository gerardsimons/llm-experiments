from pprint import pprint

import pandas as pd
from diskcache import Cache
from sklearn.metrics import classification_report
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
from tqdm import tqdm

from part3.classifiers.logprob_dynamic_classifier import LogprobDynamicFewShotClassifier

# from part3.data_loader import load_data

cache = Cache()

@cache.memoize()
def load_data(train_size=1000, test_size=100, seed=42):
    # Read data
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_table(url, header=None, names=["label", "text"])
    df["is_spam"] = (df["label"] == "spam").astype(int)

    # 1. Sample train
    train_df = df.sample(n=train_size, random_state=seed)

    # 2. Remove train rows
    remaining_df = df.drop(train_df.index)

    # 3. Sample test from remaining
    test_df = remaining_df.sample(n=test_size, random_state=seed)

    X_train = train_df['text']
    y_train = train_df['label']
    X_test = test_df['text']
    y_test = test_df['label']

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(100, 10)


# Initialise dynamicfewshotclassifier using ollama vectorizer
# Default vectorizer uses nomic-embed-text

clfs = [
    ZeroShotOllamaClassifier(model="llama3:8b"),
    # DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=1),
    DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=3),
    LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=3)
]

results = []
for clf in tqdm(clfs, desc="Fitting"):
    # clf = DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=3)

    print("Fitting data")
    clf.fit(X_train, y_train)

    print("Predicting")
    y_pred = clf.predict(X_test)
    print("y_pred:", y_pred)

    n_examples = getattr(clf, 'n_examples', 0)
    print(type(clf), clf.model, getattr(clf, 'n_examples', None))
    report = classification_report(y_test, y_pred, output_dict=True)

    f1 = report['macro avg']['f1-score']
    print(f"F1={f1}")

    results.append({
        'clf': type(clf),
        'n_examples': n_examples,
        'f1': f1
    })

df = pd.DataFrame(results).sort_values(by='f1', ascending=False)

print(df.to_csv())
    # pprint()