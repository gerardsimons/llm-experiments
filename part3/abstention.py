from pprint import pprint

from part2.logprobs import get_logprobs_cached
from part3.data_loader import load_language_data
from part3.metrics import predictive_entropy

#
X_train, y_train, X_test, y_test = load_language_data(train_size=10, test_size=0)
# print(X_train)

# Create a profile of the log probs
prompt_template = """Given the text below, determine the single most appropriate ISO 639-1 language code.
Answer with the code only.

Text: {text}

Language code:"""

def run_prompts(inputs, prompt_template, **kwargs):
    all_logprobs = []
    if isinstance(inputs, str):
        inputs = [inputs]

    for x in inputs:
        prompt = prompt_template.format(text=x)
        logprobs = get_logprobs_cached(prompt, **kwargs).logprobs
        all_logprobs.append(logprobs)

    return all_logprobs



not_language = "0103183901849103"

prompt = prompt_template.format(text=not_language)

kwargs = {"model_id": 'llama3:8b', "provider": 'ollama', "invert_log": False}
logprobs = run_prompts(not_language, prompt_template, **kwargs)

logprobs_entropy = list(map(predictive_entropy, logprobs))

print("Not language:")
print(logprobs_entropy)

logprobs = run_prompts(X_train, prompt_template, **kwargs)
logprobs_entropy = list(map(predictive_entropy, logprobs))
print("Language:")
print(logprobs_entropy)

