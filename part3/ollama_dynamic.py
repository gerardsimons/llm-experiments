from pprint import pprint

import pandas as pd
from diskcache import Cache
from sklearn.metrics import classification_report
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
from tqdm import tqdm

from part3.classifiers.logprob_dynamic_classifier import LogprobDynamicFewShotClassifier
from part3.prompts import SPAM_ZERO_SHOT_PROMPT_TEMPLATE, SPAM_FEW_SHOT_PROMPT_TEMPLATE

# from part3.data_loader import load_data

cache = Cache()

@cache.memoize()
def load_data(train_size=1000, test_size=100, seed=42):
    # Read data
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_table(url, header=None, names=["label", "text"])
    print("df.shape:", df.shape)
    df["label"] = df["label"].map({"spam": "Yes", "ham": "No"})

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

# large
# X_train, y_train, X_test, y_test = load_data(1000, 1000, seed=0)

# medium
X_train, y_train, X_test, y_test = load_data(100, 100, seed=0)

# tiny
# X_train, y_train, X_test, y_test = load_data(10, 10, seed=0)

print(y_test.value_counts())
print(y_train.value_counts())

assert len(y_train.unique()) > 1

# Initialise dynamicfewshotclassifier using ollama vectorizer
# Default vectorizer uses nomic-embed-text

n_examples = 3
clfs = [
    ZeroShotOllamaClassifier(model="llama3:8b", prompt_template=SPAM_ZERO_SHOT_PROMPT_TEMPLATE),
    # DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=1),
    DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=n_examples, prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),

    # Our example
    LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='profile_conditional', distance='l2', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
    LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='profile_conditional', distance='cosine', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
    LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='confusion_conditional', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
    LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='margin_global', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
    LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='entropy_global', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE)
]

clf_dsc = [
    'zeroshot_skollama',
    f'dynamic_skollama_{n_examples}shot',

    f'dynamic_logprob_profile_conditional_l2_{n_examples}shot',
    f'dynamic_logprob_profile_conditional_cosine_{n_examples}shot',
    f'dynamic_logprob_confusion_conditional_{n_examples}shot',
    f'dynamic_logprob_margin_global_{n_examples}shot',
    f'dynamic_logprob_entropy_global_{n_examples}shot',
]
assert len(clfs) == len(clf_dsc)

results = []
for clf, clf_description  in tqdm(zip(clfs, clf_dsc), desc="Fitting"):
    # clf = DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=3)

    print("Fitting data")
    clf.fit(X_train, y_train)

    print("Predicting")
    y_pred = clf.predict(X_test)
    print("y_pred:", y_pred)

    n_examples = getattr(clf, 'n_examples', 0)
    print(type(clf), clf.model, getattr(clf, 'n_examples', None))
    report = classification_report(y_test, y_pred, output_dict=True)

    # Macro average f1 score is most important
    f1 = report['macro avg']['f1-score']
    print(f"F1={f1}")

    results.append({
        'clf_type': type(clf),
        'clf_desc': clf_description,
        'n_examples': n_examples,
        'f1': f1,
        'report': report,
        'train_size': len(y_train),
        'test_size': len(y_test)
    })

df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
df.to_csv(f"ollama_dynamic_{len(y_train)}.csv")
df.to_pickle(f"ollama_dynamic_{len(y_train)}.pkl")
print(df.to_csv())
    # pprint()