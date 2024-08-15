from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
# client = OpenAI()
# defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# if you saved the key under a different environment variable name, you can do something like:
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poet."},
    {"role": "user", "content": "Compose a poem to greet a person."}
  ],
  temperature=0.7,  # Adjusts the creativity level of the response
  max_tokens=150,  # Limits the length of the response
  top_p=1,  # Controls the diversity via nucleus sampling
)


# print(response.choices[0].message.content)
print(response.usage.total_tokens)
print(response.usage.completion_tokens)

  


# response = ChatCompletion(
#       id='chatcmpl-9ou7vhZ7daR5Dt6YbpfMk5XREASFL', 
#       choices=[
#         Choice(
#           finish_reason='stop', 
#           index=0, 
#           logprobs=None, 
#           message=ChatCompletionMessage(
#             content="In the morning light, with colors so bright,\nI offer you a warm and heartfelt invite,\nTo embrace the day with joy and delight,\nMay your path be paved with love and insight.\n\nWelcome, dear friend, to this new dawn,\nWhere possibilities bloom and dreams spawn,\nMay happiness follow you like a faithful fawn,\nAs you journey through each day until it's gone.\n\nSo here's my greeting, sincere and true,\nA heartfelt wish just for you,\nMay your heart be light and your spirit renew,\nWelcome, dear one, to a world brand new.", 
#             role='assistant', 
#             function_call=None, 
#             tool_calls=None
#             )
#           )], 
#       created=1721919179, 
#       model='gpt-3.5-turbo-0125', 
#       object='chat.completion', 
#       service_tier=None, 
#       system_fingerprint=None, 
#       usage=CompletionUsage(
#         completion_tokens=114,
#         prompt_tokens=24, 
#         total_tokens=138
        
#     ))


# import openai
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable

# # Auto-trace LLM calls in-context
# client = wrap_openai(openai.Client())

# @traceable # Auto-trace this function
# def pipeline(user_input: str):
#     result = client.chat.completions.create(
#         messages=[{"role": "user", "content": user_input}],
#         model="gpt-3.5-turbo"
#     )
#     return result.choices[0].message.content

# pipeline("Hello, world!")
# # Out:  Hello there! How can I assist you today?
