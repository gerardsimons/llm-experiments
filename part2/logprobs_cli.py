import argparse
import math
import os
import dotenv
from pprint import pprint
import requests

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

def vertex_gemini_logprobs(prompt, top_logprobs=5, temperature=0.5, verbose=False, top_k=10, top_p=1.0, invert_log=True, model_id="gemini-2.5-flash"):

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
            max_output_tokens=10,
            thinking_config=ThinkingConfig(thinking_budget=0)
        ),
    )

    if verbose:
        pprint(response.to_json_dict())

    candidate = response.candidates[0]
    logprobs_result = candidate.logprobs_result

    result = {}
    for idx, chosen in enumerate(logprobs_result.top_candidates[0].candidates):
        prob = chosen.log_probability
        if invert_log:
            prob = math.e ** prob
        result[chosen.token] = prob

    result = sorted(
        result.items(),
        key=lambda kv: -kv[1],
    )
    return result


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
        print(f'Error getting probs from Ollama: {e}')
        return {}

def ollama_prompt_logprobs(model, prompt, invert_log=True, **kwargs):
    resp = ollama_prompt(model, prompt, **kwargs)
    if not resp:
        return {}

    result = {}
    if resp.get('logprobs') and resp['logprobs'][0].get('top_logprobs'):
        first_token_top_logprobs = resp['logprobs'][0]['top_logprobs']
        for token_prob_dict in first_token_top_logprobs:
            token = token_prob_dict['token']
            p = token_prob_dict['logprob']
            if invert_log:
                p = math.e ** p
            result[token] = p

    result = sorted(
        result.items(),
        key=lambda kv: -kv[1],
    )
    return result

# --- OpenAI Logprobs ---
import openai

def openai_logprobs(prompt, top_logprobs=10, verbose=False, temperature=0.5, invert_log=True, top_k=10, top_p=1.0, model="gpt-4.1-2025-04-14"):


    response = openai.Completion.create( # Using completions for logprobs, as chat.completions logprobs is more complex
        model=model,
        prompt=prompt,
        temperature=temperature,
        logprobs=top_logprobs, # This maps to 'top_logprobs' parameter
        max_tokens=10, # Limit output tokens for logprob analysis
    )

    result = {}
    if response.choices and response.choices[0].logprobs:
        # OpenAI returns top_logprobs for each token, but we need the overall next token logprobs
        # The structure is slightly different, we will extract the top logprobs for the first token generated
        if response.choices[0].logprobs.top_logprobs and response.choices[0].logprobs.top_logprobs[0]:
            first_token_top_logprobs = response.choices[0].logprobs.top_logprobs[0]
            for token, logprob in first_token_top_logprobs.items():
                prob = logprob
                if invert_log:
                    prob = math.e ** prob
                result[token] = prob

    result = sorted(
        result.items(),
        key=lambda kv: -kv[1],
    )

    if verbose:
        pprint(response.to_dict())
        print("\nGenerated text:")
        print(response.choices[0].text)
        print("\nToken logprobs:")
        pprint(result)

    return result

# --- CLI Setup ---
def main():
    parser = argparse.ArgumentParser(description="Get next token logprobs from various LLM providers.")
    parser.add_argument("provider", type=str, choices=["gemini", "ollama", "openai"],
                        help="The LLM provider to use.")
    parser.add_argument("prompt", type=str, help="The prompt to send to the LLM.")
    parser.add_argument("--model", type=str,
                        help="Specific model to use (e.g., for Ollama or OpenAI).")
    parser.add_argument("--top_logprobs", type=int, default=5,
                        help="Number of top log probabilities to return for the next token.")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Sampling temperature for the model.")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output including full API responses.")
    parser.add_argument("--invert_log", action="store_true",
                        help="Invert log probabilities to actual probabilities (e^logprob).")

    args = parser.parse_args()

    print(f"Using provider: {args.provider}")
    print(f"Prompt: {args.prompt}")

    logprobs_output = {}

    if args.provider == "gemini":
        if not args.model:
            parser.error("For Gemini, --model is required (e.g., gemini-2.5-flash).")
        logprobs_output = vertex_gemini_logprobs(
            prompt=args.prompt,
            top_logprobs=args.top_logprobs,
            temperature=args.temperature,
            verbose=args.verbose,
            top_k=args.top_k,
            top_p=args.top_p,
            invert_log=args.invert_log,
            model_id=args.model
        )
    elif args.provider == "ollama":
        if not args.model:
            parser.error("For Ollama, --model is required (e.g., llama3:8b).")
        logprobs_output = ollama_prompt_logprobs(
            model=args.model,
            prompt=args.prompt,
            top_logprobs=args.top_logprobs,
            temp=args.temperature,
            verbose=args.verbose,
            top_k=args.top_k,
            top_p=args.top_p,
            invert_log=args.invert_log
        )
    elif args.provider == "openai":
        if not args.model:
            parser.error("For OpenAI, --model is required (e.g., gpt-4.1-2025-04-14).")
        logprobs_output = openai_logprobs(
            prompt=args.prompt,
            top_logprobs=args.top_logprobs,
            temperature=args.temperature,
            verbose=args.verbose,
            top_k=args.top_k,
            top_p=args.top_p,
            invert_log=args.invert_log,
            model=args.model
        )
    
    if logprobs_output:
        print("\nNext Token Logprobs:")
        pprint(logprobs_output)
    else:
        print("\nCould not retrieve logprobs. Check error messages above.")

if __name__ == "__main__":
    main()
