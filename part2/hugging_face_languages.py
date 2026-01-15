import pandas as pd

splits = {'train': 'train.csv', 'validation': 'valid.csv', 'test': 'test.csv'}

# Requires optional fsspec and huggingface hub dependency; install it with pip install fsspec
df = pd.read_csv("hf://datasets/papluca/language-identification/" + splits["train"])

df.

# print(df)