from openai import OpenAI
 
client = OpenAI() # Assumes API key from env
 
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "The capital of France is"}],
    logprobs=True,
    top_logprobs=5, # Request top 5 logprobs for each token
    temperature=0, # Control randomness
    top_p=0.9 # Control nucleus sampling
)
 
# Accessing log probabilities
first_token_logprobs = response.choices[0].logprobs.content[0].top_logprobs

# This should output Paris or "the" as prelude to "the capital of France is ... "
print(first_token_logprobs)