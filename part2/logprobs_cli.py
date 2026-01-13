import argparse
import math
import os
from pprint import pprint
import requests
import openai
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig


def _get_single_logprobs(provider, prompt, model_id, **kwargs):
    """Helper to dispatch to the correct provider for a single prediction."""
    if provider == "gemini":
        return _vertex_gemini_logprobs(prompt=prompt, model_id=model_id, **kwargs)
    elif provider == "ollama":
        kwargs['temp'] = kwargs.pop('temperature', 0.5)
        return _ollama_prompt_logprobs(model=model_id, prompt=prompt, **kwargs)
    elif provider == "openai":
        return _openai_logprobs(prompt=prompt, model=model_id, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _vertex_gemini_logprobs(prompt, top_logprobs=5, temperature=0.5, verbose=False, top_k=10, top_p=1.0, invert_log=True, model_id="gemini-1.5-flash"):
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")
    response = client.models.generate_content(
        model=model_id,
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
    tokens = []
    finish_reason = candidate.finish_reason.name

    if hasattr(candidate, 'logprobs_result') and candidate.logprobs_result:
        for chosen in candidate.logprobs_result.top_candidates[0].candidates:
            prob = chosen.log_probability
            if invert_log:
                prob = math.e ** prob
            tokens.append({'token': chosen.token, 'probability': prob, 'next_tokens': []})
    
    return {'tokens': sorted(tokens, key=lambda k: -k['probability']), 'finish_reason': finish_reason}


def _ollama_prompt_logprobs(model, prompt, invert_log=True, **kwargs):
    resp = _ollama_prompt(model, prompt, **kwargs)
    if not resp:
        return {'tokens': [], 'finish_reason': 'ERROR'}
    
    tokens = []
    stop_found = False
    
    # Correctly parse the top_logprobs for the *first* token prediction
    if resp.get('logprobs') and resp['logprobs']:
        first_token_options = resp['logprobs'][0].get('top_logprobs', [])
        for token_prob_dict in first_token_options:
            token = token_prob_dict['token']
            if token == '<|eot_id|>':
                stop_found = True
                continue # Don't add the stop token to the list of choices
            prob = token_prob_dict['logprob']
            if invert_log:
                prob = math.e ** prob
            tokens.append({'token': token, 'probability': prob, 'next_tokens': []})

    finish_reason = 'STOP' if stop_found or resp.get('done_reason') == 'stop' else 'UNKNOWN'
    return {'tokens': sorted(tokens, key=lambda k: -k['probability']), 'finish_reason': finish_reason}


def _ollama_prompt(model, prompt, temp=0.5, top_k=10, top_p=1, top_logprobs=10, verbose=False):
    url = 'http://localhost:11434/api/generate'
    payload = {
        'model': model, 'prompt': prompt, 'stream': False, "temperature": temp,
        "logprobs": True, "top_k": top_k, "top_p": top_p, "top_logprobs": top_logprobs
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        resp_json = response.json()
        if verbose:
            print(f"prompt: {prompt}\nresponse:"); pprint(resp_json)
        return resp_json
    except requests.exceptions.RequestException as e:
        print(f'Error getting probs from Ollama: {e}')
        return {}


def _openai_logprobs(prompt, top_logprobs=10, verbose=False, temperature=0.5, invert_log=True, model="gpt-4.1-2025-04-14", **kwargs):
    response = openai.Completion.create(model=model, prompt=prompt, temperature=temperature, logprobs=top_logprobs, max_tokens=10)
    tokens = []
    finish_reason = 'UNKNOWN'

    if response.choices:
        choice = response.choices[0]
        finish_reason = choice.finish_reason.upper()
        if choice.logprobs and choice.logprobs.top_logprobs:
            first_token_top_logprobs = choice.logprobs.top_logprobs[0]
            for token, logprob in first_token_top_logprobs.items():
                prob = logprob
                if invert_log:
                    prob = math.e ** prob
                tokens.append({'token': token, 'probability': prob, 'next_tokens': []})
    
    if verbose:
        pprint(response.to_dict())
    return {'tokens': sorted(tokens, key=lambda k: -k['probability']), 'finish_reason': finish_reason}


def get_logprobs(provider, prompt, model_id, rollout_horizon=1, **kwargs):
    top_logprobs = kwargs.get('top_logprobs', 5)
    if rollout_horizon > 1:
        if top_logprobs > 1:
            rollout_size = int((top_logprobs**rollout_horizon - 1) / (top_logprobs - 1))
        else:
            rollout_size = rollout_horizon
        print(f"Computed rollout size: {rollout_size} API calls.")
        if rollout_size > 100:
            raise ValueError(f"Rollout size of {rollout_size} exceeds the maximum of 100. Please reduce rollout_horizon or top_logprobs.")
    
    return _recursive_logprobs(provider, prompt, model_id, rollout_horizon, **kwargs)


def _recursive_logprobs(provider, prompt, model_id, depth, **kwargs):
    if depth == 0:
        return []

    response_data = _get_single_logprobs(provider, prompt, model_id, **kwargs)
    current_level_logprobs = response_data.get('tokens', [])
    
    # Always attempt to recurse for each token if we have more depth to explore.
    # The termination of a branch is now handled by the response of the *next* call in the sequence.
    if depth > 1:
        for item in current_level_logprobs:
            # Create the prompt for the next level of the tree
            new_prompt = prompt + item['token']
            # This recursive call's return value will correctly reflect if 
            # *its* own branch should be terminated (i.e., if it returns an empty list).
            item['next_tokens'] = _recursive_logprobs(provider, new_prompt, model_id, depth - 1, **kwargs)
            
    return current_level_logprobs


def _display_as_ascii(logprobs_tree, indent=0, prefix=""):
    output = []
    for i, node in enumerate(logprobs_tree):
        is_last = (i == len(logprobs_tree) - 1)
        current_prefix = "└── " if is_last else "├── "
        line = f"{'    ' * indent}{prefix}{current_prefix}"
        
        token_display = node['token'].replace('\\n', '↵').replace('\\t', '⇥')
        line += f"'{token_display}' (p={node['probability']:.4f})"
        output.append(line)

        if node['next_tokens']:
            next_prefix = "    " if is_last else "│   "
            output.extend(_display_as_ascii(node['next_tokens'], indent + 1, prefix + next_prefix))
    return output


def _display_as_graphviz(logprobs_tree, parent_id=None, graph=None, node_id_counter=[0]):
    if graph is None:
        graph = ['digraph LogprobsTree {', '  rankdir=LR;', '  node [shape=box];']

    for node in logprobs_tree:
        current_id = node_id_counter[0]
        node_id_counter[0] += 1
        
        token_display = node['token'].replace('\\n', '\\\\n').replace('"', '\\"')
        label = f'"{token_display}\\n(p={node["probability"]:.4f})"'
        graph.append(f'  n{current_id} [label={label}];')

        if parent_id is not None:
            graph.append(f'  n{parent_id} -> n{current_id};')

        if node['next_tokens']:
            _display_as_graphviz(node['next_tokens'], current_id, graph, node_id_counter)
            
    return graph


def main():
    parser = argparse.ArgumentParser(description="Get next token logprobs from various LLM providers, with optional rollout.")
    parser.add_argument("provider", type=str, choices=["gemini", "ollama", "openai"], help="The LLM provider to use.")
    parser.add_argument("prompt", type=str, help="The prompt to send to the LLM.")
    parser.add_argument("--model_id", type=str, required=True, help="Specific model to use (e.g., 'gemini-1.5-flash', 'llama3:8b').")
    parser.add_argument("--top_logprobs", type=int, default=5, help="Number of top log probabilities to return.")
    parser.add_argument("--rollout_horizon", type=int, default=1, help="Depth of the rollout for each token.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    parser.add_argument("--no_invert_log", action="store_true", help="Do not convert log probabilities to actual probabilities.")
    parser.add_argument("--display_as", type=str, choices=["pprint", "ascii", "graphviz"], default="pprint",
                        help="Format to display the logprobs tree.")
    parser.add_argument("-o", "--output", type=str, help="Path to save the output file (for ascii or graphviz display).")


    args = parser.parse_args()

    print(f"Using provider: {args.provider}\nPrompt: '{args.prompt}'")

    try:
        logprobs_output = get_logprobs(
            provider=args.provider,
            prompt=args.prompt,
            model_id=args.model_id,
            top_logprobs=args.top_logprobs,
            rollout_horizon=args.rollout_horizon,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=args.verbose,
            invert_log=not args.no_invert_log
        )

        if logprobs_output:
            output_content = ""
            if args.display_as == "pprint":
                if args.output:
                    print("Warning: --output is ignored for 'pprint' display. Printing to console.")
                print("\n--- Token Logprobs Tree (pprint) ---")
                pprint(logprobs_output)
                
            elif args.display_as == "ascii":
                output_content = "\n".join(_display_as_ascii(logprobs_output))
                
            elif args.display_as == "graphviz":
                graph_lines = _display_as_graphviz(logprobs_output)
                graph_lines.append('}') # Close the digraph block
                output_content = "\n".join(graph_lines)

            if args.output and output_content:
                with open(args.output, 'w') as f:
                    f.write(output_content)
                print(f"\nOutput successfully saved to {args.output}")
            elif not args.output and output_content:
                 print(f"\n--- Token Logprobs Tree ({args.display_as.upper()}) ---")
                 print(output_content)

        else:
            print("\nCould not retrieve logprobs. Check error messages above.")

    except (ValueError, Exception) as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
