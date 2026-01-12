# First lets see how models perform normally, under normal conditions
import math
from pprint import pprint

import requests


def ollama_prompt(model, prompt, temp=0.5, top_k=10, top_p=1, top_logprobs=10, verbose=False):
    url = 'http://localhost:11434/api/generate'
    payload = {
        'model': model,
        'prompt': prompt,
        'stream': False,
        "temperature": temp,
        "logprobs": True,
        "top_k": top_k,
        "top_p": top_p,
        "top_logprobs": top_logprobs
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        resp_json = response.json()
        if verbose:
            print(f"prompt: {prompt}")
            print(f"response:")
            pprint(resp_json)
        return resp_json
    except requests.exceptions.RequestException as e:
        print(f'Error getting probs: {e}')

    # Note that not all models support logprobs. Ollama will not raise an error in that case, but simply omit the information from the response body!

def ollama_prompt_logprobs(model, prompt, invert_log=True, **kwargs):
    resp = ollama_prompt(model, prompt, **kwargs)

    result = {}
    top_response = resp['logprobs']
    for token_probs in top_response:
        # print(token_probs)

        token = token_probs['token']

        if invert_log:
            p = math.e ** token_probs['logprob']
            # print(token_probs['logprob'], p)
        result[token] = p

    return result

if __name__ == '__main__':
    # ollama_prompt("llama3:8b", "Please answer with 'yes' or 'no' only. The capital of the Netherlands is Amsterdam.", verbose=True)
    ollama_prompt("llama3:8b", "Please answer with 1 word only. The capital of the Netherlands is ", verbose=True)
