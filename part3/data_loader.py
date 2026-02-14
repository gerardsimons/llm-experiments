import pandas as pd
from datasets import load_dataset
from diskcache import Cache, FanoutCache
from sklearn.model_selection import train_test_split
from pathlib import Path
import kagglehub

cache = FanoutCache(".diskcache")

# No cache required since hugginface caches it
def load_language_data(train_size:int=1000, test_size:int=200):
    """
    Downloads the papluca/language-identification dataset from the Hugging Face Hub
    and returns a stratified sample for training and testing.
    This version avoids using the 'datasets' library to prevent pyarrow issues.

    Returns:
        tuple: A tuple containing X_train, y_train, X_test, y_test.
    """
    print("Loading and preparing data...")

    # Load the dataset
    dataset = load_dataset("papluca/language-identification")

    # Convert to pandas DataFrame
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # For demonstration purposes, we will use a smaller subset of the data
    train_df = train_df.sample(n=train_size, random_state=42)
    test_df = test_df.sample(n=test_size, random_state=42)

    X_train = train_df['text']
    y_train = train_df['labels']
    X_test = test_df['text']
    y_test = test_df['labels']

    print("Training data shape:", X_train.shape)
    print("Test data shape:", X_test.shape)
    print("Sample training data:")
    print(X_train.head())
    print("Sample training labels:")
    print(y_train.head())
    
    return X_train, y_train, X_test, y_test



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
    # import kagglehub # Already imported at the top now

    # Download latest version
    path = kagglehub.dataset_download("amananandrai/ag-news-classification-dataset")

    print("Path to dataset files:", path)

    # Load full datasets first
    full_train_df = pd.read_csv(Path(path) / 'train.csv')
    full_test_df = pd.read_csv(Path(path) / 'test.csv')

    real_labels = ['World', 'Sports', 'Business', 'Tech/Sci']

    full_train_df['label'] = full_train_df['Class Index'].map(lambda x: real_labels[x - 1][0].upper())
    full_test_df['label'] = full_test_df['Class Index'].map(lambda x: real_labels[x - 1][0].upper())

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

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_language_data()
    print("\nSample Training Data:")
    print(X_train.head())
    print("\nSample Training Labels:")
    print(y_train.head())
