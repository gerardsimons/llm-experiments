import sys
import time
from pathlib import Path
from pprint import pprint

import pandas as pd
from diskcache import Cache
from sklearn.metrics import classification_report
from skollama.models.ollama.classification.few_shot import DynamicFewShotOllamaClassifier
from skollama.models.ollama.classification.zero_shot import ZeroShotOllamaClassifier
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from part3.classifiers.logprob_dynamic_classifier import LogprobDynamicFewShotClassifier
from part3.prompts import NEWS_FEW_SHOT_PROMPT, NEWS_ZERO_SHOT_PROMPT


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

def load_ag_news_data(train_size, test_size, seed=42):
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

    print("Path to dataset files:", path)

    # Load full datasets first
    full_train_df = pd.read_csv(Path(path) / 'train.csv')
    full_test_df = pd.read_csv(Path(path) / 'test.csv')

    real_labels = ['World', 'Sports', 'Business', 'Tech/Sci']

    full_train_df['label'] = full_train_df['Class Index'].map(lambda x : real_labels[x-1][0].upper())
    full_test_df['label'] = full_test_df['Class Index'].map(lambda x: real_labels[x-1][0].upper())

    # Stratified sampling for training set
    if train_size >= len(full_train_df):
        train_df = full_train_df
    else:
        train_df, _ = train_test_split(
            full_train_df,
            train_size=train_size,
            stratify=full_train_df['label'],
            random_state=seed
        )

    # Stratified sampling for testing set
    if test_size >= len(full_test_df):
        test_df = full_test_df
    else:
        test_df, _ = train_test_split(
            full_test_df,
            train_size=test_size,
            stratify=full_test_df['label'],
            random_state=seed
        )

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
    elif isinstance(model, DynamicFewShotOllamaClassifier):
        return f"dynamic_skollama_{model.n_examples}"

def run_experiments(dl_func, train_size, test_size, seed=0, tag="", n_examples_all=[3]):
    start_t = time.time()
    print(f"Running experiments {train_size=} {test_size=} {seed=} {tag=}")
    X_train, y_train, X_test, y_test = dl_func(train_size, test_size, seed=seed)

    prompt_templates = {
        'few': NEWS_FEW_SHOT_PROMPT,
        'zero': NEWS_ZERO_SHOT_PROMPT,
    }

    results = []
    experiment_id = 0

    for n_examples in n_examples_all:
        models = [
            ZeroShotOllamaClassifier(model="llama3:8b", prompt_template=prompt_templates['zero']),
            DynamicFewShotOllamaClassifier(model="llama3:8b", n_examples=n_examples, prompt_template=prompt_templates['few']),
            LogprobDynamicFewShotClassifier(prompt_templates=prompt_templates, model="llama3:8b", n_examples=n_examples, strategy='random', verbose=True),
            LogprobDynamicFewShotClassifier(prompt_templates=prompt_templates, model="llama3:8b", n_examples=n_examples, strategy='profile_conditional', distance='l2', verbose=True),
            LogprobDynamicFewShotClassifier(prompt_templates=prompt_templates, model="llama3:8b", n_examples=n_examples, strategy='profile_conditional', distance='cosine', verbose=True),
            LogprobDynamicFewShotClassifier(prompt_templates=prompt_templates, model="llama3:8b", n_examples=n_examples, strategy='confusion_conditional', verbose=True),
            LogprobDynamicFewShotClassifier(prompt_templates=prompt_templates, model="llama3:8b", n_examples=n_examples, strategy='margin_global', verbose=True),
            LogprobDynamicFewShotClassifier(prompt_templates=prompt_templates, model="llama3:8b", n_examples=n_examples, strategy='entropy_global', verbose=True)
        ]

        for model  in tqdm(models, desc="Fitting"):
            exp_t = time.time()
            model_desc = model_description_str(model)

            print(f"Model={model_desc}")
            print("Fitting data")
            model.fit(X_train, y_train)

            print(f"Predicting {len(X_test)}")
            y_pred = model.predict(X_test)
            # print("y_pred:", y_pred)

            n_examples = getattr(model, 'n_examples', 0)
            # print(type(clf), clf.model, getattr(clf, 'n_examples', None))
            report = classification_report(y_test, y_pred, output_dict=True)

            # Macro average f1 score is most important
            f1 = report['macro avg']['f1-score']


            print(f"Experiment #{experiment_id} Finished ({model_desc})")
            print(f"F1={f1}")
            elapsed = time.time() - exp_t
            results.append({
                'experiment_id': experiment_id,
                'clf_type': type(model),
                'clf_desc': model_desc,
                'n_examples': n_examples,
                'f1': f1,
                'report': report,
                'tag': tag,
                'train_size': len(y_train),
                'test_size': len(y_test),
                'seed': seed,
                'duration': elapsed
            })
            experiment_id += 1

    elapsed = time.time() - start_t
    print(f"Elapsed:{elapsed:.3f}")
    df = pd.DataFrame(results).sort_values(by='f1', ascending=False)
    fname = f"ollama_dynamic_{len(y_train)}_{tag}"
    df.to_csv(f"{fname}.csv")
    df.to_pickle(f"{fname}.pkl")
    print(f"Saved to {fname}.{{.csv,.pkl}}")
    return df

# def

if __name__ == '__main__':

    # sys.exit()
    # large
    # train_size = 1000
    # test_size = 100
    # n_examples_all = [1, 3, 10]

    # medium
    train_size = 100
    test_size = 100
    n_examples_all = [3]

    # tiny
    # train_size = 10
    # test_size = 10
    # n_examples_all = [1]

    tag = "agnews"

    run_experiments(load_ag_news_data, train_size, test_size, tag=tag, n_examples_all=n_examples_all)

    # pprint()