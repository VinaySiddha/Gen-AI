import gradio as gr
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI

# Setup RAG model
def setup_rag_model():
    loader = PDFMinerLoader("sample.pdf")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    docs = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever()
    
    llm = OpenAI(temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, return_source_documents=True)
    
    return qa_chain

# Initialize RAG model
qa_chain = setup_rag_model()

def answer_query(query):
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]

# Create Gradio interface
interface = gr.Interface(
    fn=answer_query,
    inputs=gr.Textbox(label="Enter your query"),
    outputs=[gr.Textbox(label="Answer"), gr.JSON(label="Source Documents")],
    title="RAG Model Query Interface",
    description="Enter your query to get answers from the RAG model."
)

if __name__ == "__main__":
    interface.launch()