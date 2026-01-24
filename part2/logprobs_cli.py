import argparse
from pprint import pprint

from part2.logprobs import get_logprobs


def main():
    parser = argparse.ArgumentParser(description="Get next token logprobs from various LLM providers.")
    parser.add_argument("provider", type=str, choices=["gemini", "ollama", "openai"], help="The LLM provider to use.")
    parser.add_argument("prompt", type=str, help="The prompt to send to the LLM.")
    parser.add_argument("--model_id", type=str, required=True, help="Specific model to use (e.g., 'gemini-1.5-flash', 'llama3:8b', 'gpt-4.1-2025-04-14').")
    parser.add_argument("--top_logprobs", type=int, default=5, help="Number of top log probabilities to return for the next token.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature for the model.")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling parameter.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output including full API responses.")
    parser.add_argument("--no_invert_log", action="store_true", help="Do not invert log probabilities to actual probabilities.")
    parser.add_argument("--json", dest='to_json', action="store_true", help="Output as JSON.")

    args = parser.parse_args()
    
    if not args.to_json:
      print(f"Using provider: {args.provider}")
      print(f"Prompt: {args.prompt}")

    try:
        logprobs_result = get_logprobs(
            provider=args.provider,
            prompt=args.prompt,
            model_id=args.model_id,
            top_logprobs=args.top_logprobs,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            verbose=args.verbose,
            invert_log=not args.no_invert_log
        )

        if logprobs_result:
            if args.to_json:
                print(logprobs_result.to_json())
            else:
                print("\nNext Token Logprobs:")
                pprint(logprobs_result.logprobs, sort_dicts=False)
                print(f"\nFull response: {logprobs_result.response_text}")
        else:
            print("\nCould not retrieve logprobs. Check error messages above.")

    except (ValueError, Exception) as e:
        print(f"Error: {e}")



if __name__ == "__main__":
    main()
