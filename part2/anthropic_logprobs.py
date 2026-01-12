from pprint import pprint
import os
import dotenv
import anthropic

dotenv.load_dotenv()
print(os.getenv("ANTHROPIC_API_KEY")[:8] + "******")

top_logprobs = 5


def claude_logprobs(
    prompt,
    top_logprobs=5,
    verbose=False,
    temperature=0.5,
):
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-3-5-sonnet-2024-10-22",
        max_tokens=5,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        logprobs=True,
        top_k=top_logprobs,
    )

    # Claude returns content as a list of blocks
    text_block = response.content[0]

    result = {
        "text": text_block.text,
        "logprobs": text_block.logprobs,
    }

    if verbose:
        pprint(response.model_dump())
        print("\nGenerated text:")
        print(text_block.text)

        print("\nToken logprobs:")
        for token_info in text_block.logprobs["tokens"]:
            print(
                f"Token: {token_info['token']!r} | "
                f"logprob: {token_info['logprob']:.4f}"
            )

            if token_info.get("top_logprobs"):
                print("  Alternatives:")
                for alt in token_info["top_logprobs"]:
                    print(
                        f"    {alt['token']!r}: {alt['logprob']:.4f}"
                    )

    return result


if __name__ == "__main__":
    prompt = (
        "Please answer with 'yes' or 'no' only. "
        "The capital of the Netherlands is Amsterdam."
    )

    claude_logprobs(prompt, verbose=True)
