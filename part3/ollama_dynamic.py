import sys
from pathlib import Path
from pprint import pprint

import pandas as pd
from diskcache import Cache
from sklearn.metrics import classification_report
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
from tqdm import tqdm

from part3.classifiers.logprob_dynamic_classifier import LogprobDynamicFewShotClassifier
from part3.prompts import SPAM_ZERO_SHOT_PROMPT_TEMPLATE, SPAM_FEW_SHOT_PROMPT_TEMPLATE

cache = Cache()

@cache.memoize()
def load_spam_data(train_size=1000, test_size=100, seed=42):
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

def load_ag_news_data():#train_size, test_size, model_descriptions, models, seed=0, n_examples=3, tag=""):
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

    print("Path to dataset files:", path)

    train_df = pd.read_csv(Path(path) / 'train.csv')
    test_df = pd.read_csv(Path(path) / 'test.csv')

    real_labels = ['World', 'Sports', 'Business', 'Sci/Tech']

    train_df['label'] = train_df['Class Index'].map(lambda x : real_labels[x-1])
    test_df['label'] = test_df['Class Index'].map(lambda x: real_labels[x-1])

    # There's also description ... add that later?
    X_train = train_df['Title']
    y_train = train_df['label']
    X_test = test_df['Title']
    y_test = test_df['label']

    return X_train, y_train, X_test, y_test

def model_description_str(model):
    if isinstance(model, ZeroShotOllamaClassifier):
        return "zeroshot_skollama"
    elif isinstance(model, LogprobDynamicFewShotClassifier):
        # dynamic_logprob_profile_conditional_l2_
        if 'global' in model.strategy or 'random' in model.strategy:
            return f"dynamic_logprob_{model.strategy}"
        else:
            return f"dynamic_logprob_{model.strategy}_{model.distance}"

def run_experiments(train_size, test_size, seed=0, tag="", n_examples_all=[3]):
    X_train, y_train, X_test, y_test = load_spam_data(train_size, test_size, seed=seed)

    results = []

    for n_examples in n_examples_all:
        models = [
            # ZeroShotOllamaClassifier(model="llama3:8b", prompt_template=SPAM_ZERO_SHOT_PROMPT_TEMPLATE),
            # DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=n_examples, prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),

            LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE, strategy='random'),

            # Our examples
            # LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='profile_conditional', distance='l2',
            #                                 prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
            LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='profile_conditional', distance='cosine',
                                            prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
            # LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='confusion_conditional',
            #                                 prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
            # LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='margin_global', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE),
            # LogprobDynamicFewShotClassifier(model="llama3:8b", n_examples=n_examples, strategy='entropy_global', prompt_template=SPAM_FEW_SHOT_PROMPT_TEMPLATE)
        ]

        for model  in tqdm(models, desc="Fitting"):
            model_desc = model_description_str(model)

            print("Fitting data")
            model.fit(X_train, y_train)

            print("Predicting")
            y_pred = model.predict(X_test)
            # print("y_pred:", y_pred)

            n_examples = getattr(model, 'n_examples', 0)
            # print(type(clf), clf.model, getattr(clf, 'n_examples', None))
            report = classification_report(y_test, y_pred, output_dict=True)

            # Macro average f1 score is most important
            f1 = report['macro avg']['f1-score']

            print("Experiment Finished")
            print(f"Model={model_desc}")
            print(f"F1={f1}")

            results.append({
                'clf_type': type(model),
                'clf_desc': model_desc,
                'n_examples': n_examples,
                'f1': f1,
                'report': report,
                'tag': tag,
                'train_size': len(y_train),
                'test_size': len(y_test),
                'seed': seed
            })



    df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
    fname = f"ollama_dynamic_{len(y_train)}_{tag}"
    df.to_csv(f"{fname}.csv")
    df.to_pickle(f"{fname}.pkl")
    print(f"Saved to {fname}.{{.csv,.pkl}}")
    return df

# def

if __name__ == '__main__':
    load_ag_news_data()

    # sys.exit()
    # large
    # train_size = 1000
    # test_size = 1000

    # medium
    train_size = 100
    test_size = 1000
    n_examples_all = [3]

    # tiny
    # train_size = 10
    # test_size = 100
    # n_examples_all = [1]

    tag = "agnews"

    run_experiments(train_size, test_size, tag=tag, n_examples_all=n_examples_all)

    # pprint()