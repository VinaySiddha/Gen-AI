import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Access your API key as an environment variable.
genai.configure(api_key=os.environ.get('API_KEY'))


# Choose a model that's appropriate for your use case.
model = genai.GenerativeModel('gemini-1.5-flash')

prompt = "How are you?"

response = model.generate_content(prompt)

print(response.usage_metadata.total_token_count) #total token_count
# print(response.parts[0].text) #output
# print(response.candidates[0].content.parts[0].text) #output
# print(response.text) #output
# print(response)

# ##response

# response = GenerateContentResponse(
#     done=True,
#     iterator=None,
#     result=protos.GenerateContentResponse({
#       "candidates": [
#         {
#           "content": {
#             "parts": [
#               {
#                 "text": "As an AI, I don't have feelings or experiences like humans do, so I don't feel \"good\" or \"bad.\" I'm here to assist you with any questions or tasks you have! How can I help you today? \n"
#               }
#             ],
#             "role": "model"
#           },
#           "finish_reason": "STOP",
#           "index": 0,
#           "safety_ratings": [
#             {
#               "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
#               "probability": "NEGLIGIBLE"
#             },
#             {
#               "category": "HARM_CATEGORY_HATE_SPEECH",
#               "probability": "NEGLIGIBLE"
#             },
#             {
#               "category": "HARM_CATEGORY_HARASSMENT",
#               "probability": "NEGLIGIBLE"
#             },
#             {
#               "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
#               "probability": "NEGLIGIBLE"
#             }
#           ]
#         }
#       ],
#       "usage_metadata": {
#         "prompt_token_count": 5,
#         "candidates_token_count": 51,
#         "total_token_count": 56
#       }
#     }
                        