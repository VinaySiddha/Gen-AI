from openai import OpenAI
from dotenv import load_dotenv
import os


load_dotenv()
client = OpenAI(
  api_key = os.environ.get("OPENAI_API_KEY"),
)

def Prompt():
  while(1):
    prompt = input("User: ")
    key_word(prompt)
  
def key_word(prompt):
    if prompt == "exit":
      exit
    else:
      print(engine(prompt))

def engine(prompt):
  response = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
          {"role": "system", "content": "You are a teacher"},
          {"role": "user", "content": prompt}
        ],
        temperature=0.7,  # Adjusts the creativity level of the response
        max_tokens=150,  # Limits the length of the response
        top_p=1,  # Controls the diversity via nucleus sampling
        )
  return response.choices[0].message.content

if __name__ == "__main__":
  Prompt()


