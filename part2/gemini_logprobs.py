import math
from pprint import pprint
import os
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
MODEL_ID = "gemini-2.5-flash"  # or another version available in your project

def vertex_gemini_logprobs(prompt, top_logprobs=5, temperature=0.5, verbose=False, top_k=10, top_p=1.0, invert_log=True):
    # Initialize Vertex AI GenAI client
    client = genai.Client(vertexai=True, project=PROJECT_ID, location="global")

    # Generate content with logprobs enabled
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,
            response_logprobs=True,
            logprobs=top_logprobs,
            top_k=top_k,
            top_p=top_p,
            max_output_tokens=10,  # limit output length to necessary tokens only
            thinking_config=ThinkingConfig(thinking_budget=0) # Otherwise we get thinking tokens!
        ),
        # response_mime_type="application/json",  # structured output
    )

    if verbose:
        # Print full response for inspection
        pprint(response.to_json_dict())

    # Access the first candidate
    candidate = response.candidates[0]
    logprobs_result = candidate.logprobs_result

    # Print chosen tokens and their logprobs
    # print("Generated Text:\n", candidate.output_text)
    result = {}
    print("\nToken Logprobs:")
    for idx, chosen in enumerate(logprobs_result.top_candidates[0].candidates):
    # for idx, chosen in enumerate(logprobs_result.chosen_candidates):
        prob = chosen.log_probability
        if invert_log:
            prob = math.e ** prob
        result[chosen.token] = prob

    # after building `result`
    result = sorted(
        result.items(),
        key=lambda kv: -kv[1], # sort by probability, note - for inverting
    )

    return result

if __name__ == "__main__":
    # prompt = "Please answer with 'yes' or 'no' only. The capital of the Netherlands is Amsterdam."
    # prompt = "Please answer with 1 word only. The capital of the Netherlands is The Ha"
    prompt = "Please complete the following sentence with 1 word only. The seat of government in the Netherlands is located in the city of The H"
    probs = vertex_gemini_logprobs(prompt, verbose=True)

    pprint(probs)
