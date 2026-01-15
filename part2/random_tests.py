from pprint import pprint

from part2.logprobs_cli import get_logprobs

model_id = "llama3:8b"

provider = "ollama"
# allowed_iso_codes = ["en", "fr", "de", "nl", "lim"]
allowed_iso_codes = ["en", "fr", "de", "nl"]
allowed_iso_codes_str = [f'{x}' for x in allowed_iso_codes]
sentences = [
    "I am going home today.",  # English
    "Je rentre chez moi aujourd’hui.",  # French
    "Ich gehe heute nach Hause.",  # German
    "Ik ga vandaag naar huis.",  # Dutch (NL)
    # "Ich gaon vandage nao hoes.",  # Limburgish (Netherlands)
    # "Ich goan vandage noa hoes.",  # Limburgish (Belgium)
    # "Isch jonn hück noh Huus."  # Limburgish (Germany)
    "我今天回家。", # Chinese (Simplified, Standard Mandarin):
    "Brano lisket vande moru naa hest.", # Gibberish
    "iaUox6QW3GwNik2N" # Random -> English prior
]

prompt_template = """
You are performing language identification.

Task:
Given the text below, determine the single most appropriate ISO 639 language code.

Rules:
- Choose exactly one code from the list.
- Do not explain your answer.
- Do not output anything except the ISO code.
- Do not output any punctuation marks.

Allowed ISO 639-3 codes: {allowed_iso_codes_str}

Text:
{text}

Language code:
"""

txt = sentences[-1]
print("Text:", txt)
prompt = prompt_template.format(allowed_iso_codes_str=allowed_iso_codes_str, text=txt)
# print("Prompt: ", prompt)
logprob_kwargs = {'temperature': 0}

result = get_logprobs(
    model_id=model_id,
    provider=provider,
    prompt=prompt,
    **logprob_kwargs
)

pprint(result.logprobs)
