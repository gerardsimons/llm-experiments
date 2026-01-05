from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
print(os.getenv("OPENAI_API_KEY")[:8] + "******")
    
client = OpenAI()

response = client.responses.create(
    model="gpt-5-nano",
    input="Write a one-sentence bedtime story about a unicorn.",
    include=["message.output_text.logprobs"],
    top_logprobs=5
)
print(response)
print(response.output_text)