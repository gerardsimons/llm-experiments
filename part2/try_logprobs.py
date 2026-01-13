import os
import random
import time
from datetime import datetime
from pprint import pprint

import pandas as pd

from part2.logprobs_cli import get_logprobs

num_sessions = 3
session_size = 3

# num_sessions = 10
# num_repeats = 100

# provider='gemini'
# model_id='gemini-2.5-flash'

provider='ollama'
model_id='llama3:8b'

os.makedirs('results/', exist_ok=True)

df_rows = []

logprob_kwargs = {
    'top_logprobs': 10,
    'temperature': 0,
    'top_k': 1
}

providers = ["ollama", 'ollama', 'gemini', 'openai']
model_ids = ["llama3:8b", 'dolphin-mistral:latest', 'gemini-2.5-flash', 'gpt-4o-mini']
# prompt = "Please provide the numerical only. Give me a random number between 0 and 10."
prompt = "Return exactly one integer between 0 and 10 inclusive. Output only the integer. Do not include any text, symbols, whitespace, or formatting. Any other output is invalid."
start_dt = datetime.now()
for model_id, provider in zip(model_ids, providers):
    for sess_id in range(num_sessions):
        for sess_round in range(session_size):
            print(f"Model={model_id} Session #{sess_id} Round={sess_round}")
            dt = datetime.now()

            # print(f"Asking {model_id}")
            result = get_logprobs(
                model_id=model_id,
                provider=provider,
                prompt=prompt,
                # prompt="The city I am thinking of is",
                **logprob_kwargs
            )
            # print("Datetime:")
            # print(dt)

            print("LogProbs:")
            pprint(result.logprobs)

            print("Full Response:")
            pprint(result.response_text)

            # Wait random time to increase entropy
            wait_time = random.random() + 0.05
            print(f"Wait for {wait_time:.2f} seconds.")
            time.sleep(wait_time)

            df_rows.append({
                'session_id': sess_id,
                'session_round': sess_round,
                'timestamp': dt,
                'model_id':model_id,
                'provider': provider,
                'logprob_kwargs': logprob_kwargs,
                'prompt': dt,
                'logprob_dict': dict(result.logprobs),
                'full_response': result.response_text
            })

        print(f"End of session #{sess_id}")
        time.sleep(random.random() * 10)

df = pd.DataFrame(df_rows)
start_dt_suffix = start_dt.strftime("%Y%m%d_%H%M%S")
print(f"Saving results to {start_dt_suffix}")
df.to_csv(start_dt_suffix)
print(df.to_csv())
    # df.sort_values()