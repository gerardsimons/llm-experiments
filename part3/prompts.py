# Central repository for all prompt templates

# ==============================================================================
# News Classification Prompts
# ==============================================================================

NEWS_ZERO_SHOT_PROMPT = """You are an expert news classifier. Your task is to classify the news headline into one of the following categories:
- World (W)
- Sports (S)
- Business (B)
- Science/Technology (T)

The available labels are: {labels}.

Please provide only the single-letter code for the category.

Headline:
```{x}```

Category Code:
"""

NEWS_FEW_SHOT_PROMPT = """You are an expert news classifier. Your task is to classify the news headline into one of the following categories:
- World (W)
- Sports (S)
- Business (B)
- Science/Technology (T)

The available labels are: {labels}.

Here are some examples of correctly classified headlines:
{training_data}

Now, classify the following headline. Please provide only the single-letter code for the category.

Headline:
```{x}```

Category Code:
"""

news_prompts = {
    'few': NEWS_FEW_SHOT_PROMPT,
    'zero': NEWS_ZERO_SHOT_PROMPT
}

SPAM_ZERO_SHOT_PROMPT_V2 = """You are classifying SMS messages.

    Question: Is the following SMS message spam? The possible labels are: {labels}

    Answer with exactly one token: Yes or No.

    Message:
    `{x}`

    Answer:
    """