import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
from docx import Document

# Load OpenAI API key
load_dotenv()


client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY"),
)

cli = OpenAI()


# openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to create embeddings using OpenAI's embedding model
def create_embeddings(text):
    response = cli.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    # Access the embedding using the correct properties
    embedding = response.data[0].embedding
    return np.array(embedding)

# Compare the query with the document
def compare_embeddings(document_embedding, query_embedding):
    # Calculate the cosine similarity between the document and the query
    cosine_similarity = np.dot(document_embedding, query_embedding) / (np.linalg.norm(document_embedding) * np.linalg.norm(query_embedding))
    return cosine_similarity

# Ask questions and get answers
def ask_question(document_text, question):
    document_embedding = create_embeddings(document_text)
    query_embedding = create_embeddings(question)
    
    # Calculate similarity between document and query
    similarity = compare_embeddings(document_embedding, query_embedding)
    
    # If similarity is above a threshold, use the document to answer the question
    if similarity > 0.2:  # You can adjust the threshold
        prompt = f"The following is a document:\n\n{document_text}\n\nAnswer the following question based on the document:\n{question}"
        response = client.chat.completions.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    else:
       return "The document is not relevant to the question."

# Function to read text from a Word document
def read_word_document(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)

# Example usage
if __name__ == "__main__":
    # Read the document content from a Word file
    document_path = "cvis.docx"  # Replace with the actual file path
    document_text = read_word_document(document_path)

    # Provide the question you want to ask
    question = "what is computer vision?"
    
    # Call the function to get the answer
    answer = ask_question(document_text, question)
    
    # Print the answer
    print("Answer:", answer)


# openai.PermissionDeniedError: Error code: 403 - 
# {
#     'error': 
#         {
#             'message': 'You are not allowed to generate embeddings from this model', 
#             'type': 'invalid_request_error', 'param': None, 'code': None
#             }
#         }
# }