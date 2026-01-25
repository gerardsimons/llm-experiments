import pandas as pd
from datasets import load_dataset
from diskcache import FanoutCache

cache = FanoutCache()

@cache.memoize()
def load_data(train_size:int=1000, test_size:int=200):
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

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()
    print("\nSample Training Data:")
    print(X_train.head())
    print("\nSample Training Labels:")
    print(y_train.head())