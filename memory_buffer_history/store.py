from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def prompt():
    while True:
        user_prompt = input("User: ")
        if user_prompt.lower() == "exit":
            break
        response = engine(user_prompt)
        print("Assistant:", response)

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
    output = response.choices[0].message.content.strip()
    store(prompt, output)
    return output

def store(prompt, output):
    thread = {"prompt": prompt, "output": output}
    print(thread)

if __name__ == "__main__":
    prompt()
