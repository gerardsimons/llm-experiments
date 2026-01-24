
# Central repository for all prompt templates

# ==============================================================================
# Scikit-LLM / Scikit-Ollama Prompts
# ==============================================================================

# Default is often internal, but we can define our own to override.
# Note: The actual prompt construction in scikit-llm is more complex, involving
# dynamic examples for few-shot. This is a simplified view. We will inject
# this into our custom classifier.

SKLLM_ZERO_SHOT_PROMPT = """
You are a text classification assistant. Your task is to classify the given text into one of the following labels:
{labels}

Provide only the label as your response.

Text:
```{text}```

Label:
"""

SKLLM_FEW_SHOT_PROMPT = """
You are a text classification assistant. Your task is to classify the given text into one of the following labels:
{labels}

Here are some examples of correctly classified texts:
{examples}

Now, classify the following text. Provide only the label as your response.

Text:
```{text}```

Label:
"""


# ==============================================================================
# LangChain Prompts
# ==============================================================================

LANGCHAIN_PROMPT_TEMPLATE = """
You are an expert in language identification. Your task is to identify the language of the provided text.
The possible languages are:
{labels}

Please classify the following text. Respond with only the two-letter language code (e.g., 'en', 'fr', 'es').

Text:
"{text}"

Language Code:
"""


# ==============================================================================
# LlamaIndex Prompts
# ==============================================================================

LLAMAINDEX_PROMPT_TEMPLATE = """
Identify the language of the following text. The possible languages are:
{labels}

Text:
"{text}"
"""

# ==============================================================================
# Instructor Prompts
# ==============================================================================

INSTRUCTOR_PROMPT_TEMPLATE = """
Identify the language of the following text.
Text: {text}
"""
