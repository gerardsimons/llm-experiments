# Caching helper functions
import math
import os
import requests
from openai import OpenAI
from dataclasses import dataclass, asdict
from pprint import pprint
from typing import Any, OrderedDict

from diskcache import Cache
import hashlib
import json

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

cache = Cache("./logprob_cache")


@dataclass
class LogProbsResult:
    response_text: str
    logprobs: dict
    raw_response: Any

    def to_json(self):
        return json.dumps(asdict(self), indent=4)


def _vertex_gemini_logprobs(prompt, top_logprobs=5, temperature=0.5, verbose=False, top_k=10, top_p=1.0, invert_log=True, model_id="gemini-1.5-flash",
                            max_output_tokens=10):
    if verbose:
        print("Vertex Gemini LogProbs Args:")
        print(f"{prompt=}")
        print(f"{top_logprobs=}")
        print(f"{temperature=}")
        print(f"{top_k=}")
        print(f"{top_p=}")
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    MODEL_ID = model_id
    client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,
            response_logprobs=True,
            logprobs=top_logprobs,
            top_k=top_k,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            thinking_config=ThinkingConfig(thinking_budget=0)  # Never do any thinking
        ),
    )
    response_dict = response.to_json_dict()
    if verbose:
        pprint(response_dict)
    candidate = response.candidates[0]
    logprobs_result = candidate.logprobs_result
    result = {}
    for idx, chosen in enumerate(logprobs_result.top_candidates[0].candidates):
        prob = chosen.log_probability
        if invert_log:
            prob = math.e ** prob
        result[chosen.token] = prob
    # Sort the items by probability (descending) and create an ordered dictionary
    sorted_items = sorted(result.items(), key=lambda kv: -kv[1])
    logprobs = OrderedDict(sorted_items)
    return LogProbsResult(
        response_text=candidate.content.parts[0].text,
        logprobs=logprobs,
        raw_response=response_dict
    )


def _ollama_prompt(model, prompt, temp=0.5, top_k=10, top_p=1, top_logprobs=10, verbose=False):
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
            print("request:")
            pprint(payload)
            print("response:")
            pprint(resp_json)
        return resp_json
    except requests.exceptions.RequestException as e:
        print(f"Error getting probs from Ollama: {e}")
        print("Problematic Payload:")
        pprint(payload)
        raise e


def _ollama_prompt_logprobs(model, prompt, invert_log=True, **kwargs):
    resp = _ollama_prompt(model, prompt, **kwargs)
    if not resp:
        return None
    result = {}
    if resp.get('logprobs') and resp['logprobs'] and resp['logprobs'][0].get('top_logprobs'):
        first_token_top_logprobs = resp['logprobs'][0]['top_logprobs']
        for token_prob_dict in first_token_top_logprobs:
            token = token_prob_dict['token']
            p = token_prob_dict['logprob']
            if invert_log:
                p = math.e ** p
            result[token] = p
    # Sort the items by probability (descending) and create an ordered dictionary
    sorted_items = sorted(result.items(), key=lambda kv: -kv[1])
    logprobs = OrderedDict(sorted_items)
    return LogProbsResult(
        response_text=resp.get('response', ''),
        logprobs=logprobs,
        raw_response=resp
    )


def _openai_logprobs(prompt, top_logprobs=10, verbose=False, temperature=0.5, invert_log=True, top_p=1.0, model="gpt-4o-mini"):
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
        max_tokens=10,
        top_p=top_p,
    )
    result = {}
    if response.choices and response.choices[0].logprobs and response.choices[0].logprobs.content:
        # We are interested in the logprobs of the first generated token
        first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        for token_data in first_token_logprobs:
            prob = token_data.logprob
            if invert_log:
                prob = math.e ** prob
            result[token_data.token] = prob

    # Sort the items by probability (descending) and create an ordered dictionary
    sorted_items = sorted(result.items(), key=lambda kv: -kv[1])
    logprobs = OrderedDict(sorted_items)

    response_dict = response.model_dump()
    if verbose:
        pprint(response_dict)
        print("\nGenerated text:")
        print(response.choices[0].message.content)
        print("\nToken logprobs:")
        pprint(logprobs, sort_dicts=False)

    return LogProbsResult(
        response_text=response.choices[0].message.content,
        logprobs=logprobs,
        raw_response=response_dict
    )


def get_logprobs(
        provider,
        prompt,
        model_id,
        top_logprobs=10,
        temperature=0,
        top_k=10,
        top_p=1.0,
        verbose=False,
        invert_log=True
) -> LogProbsResult | None:
    if provider == "gemini":
        return _vertex_gemini_logprobs(
            prompt=prompt,
            top_logprobs=top_logprobs,
            temperature=temperature,
            verbose=verbose,
            top_k=top_k,
            top_p=top_p,
            invert_log=invert_log,
            model_id=model_id
        )
    elif provider == "ollama":
        return _ollama_prompt_logprobs(
            model=model_id,
            prompt=prompt,
            top_logprobs=top_logprobs,
            temp=temperature,
            verbose=verbose,
            top_k=top_k,
            top_p=top_p,
            invert_log=invert_log
        )
    elif provider == "openai":
        return _openai_logprobs(
            prompt=prompt,
            top_logprobs=top_logprobs,
            temperature=temperature,
            verbose=verbose,
            top_p=top_p,
            invert_log=invert_log,
            model=model_id
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def _make_cache_key(prompt: str, logprob_kwargs: dict) -> str:
    payload = {
        "prompt": prompt,
        "kwargs": logprob_kwargs,
    }
    dumped = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


def get_logprobs_cached(prompt: str, **logprob_kwargs):
    key = _make_cache_key(prompt, logprob_kwargs)

    if key in cache:
        return cache[key]

    res = get_logprobs(prompt=prompt, **logprob_kwargs)
    cache[key] = res
    return res
