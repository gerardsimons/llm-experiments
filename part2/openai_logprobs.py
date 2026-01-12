from pprint import pprint

from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY")[:8] + "******")

top_logprobs = 5


def openai_logprobs(prompt, top_logprobs=10, verbose=False, temperature=0.5, invert_log=True, top_k=10, top_p=1.0):
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4.1-2025-04-14",
        input=prompt,
        temperature=temperature,
        top_logprobs=top_logprobs,
        top_k=top_k,
        top_p=top_p,
        include=["message.output_text.logprobs"],
    )

    result = {}

    if verbose:
        pprint(response.to_dict())
        print(response.output_text)

if __name__ == '__main__':
    # prompt = "Please answer with 1 word only. The capital of the Netherlands is "
    prompt = "Please answer with 'yes' or 'no' only. The capital of the Netherlands is Amsterdam."
    openai_logprobs(prompt, verbose=True)
