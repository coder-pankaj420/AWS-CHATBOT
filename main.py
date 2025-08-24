# main.py

# --- Import necessary tools ---
import os
from fastapi import FastAPI
from pydantic import BaseModel

# --- Import LangChain and Pinecone tools ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings # Use the new, correct import
from langchain_pinecone import PineconeVectorStore

# --- CONFIGURATION BLOCK ---
# Set all keys needed for the entire project here
os.environ["PINECONE_API_KEY"] = "pcsk_3iQo4y_HMCYuMamhX1dzF74Niw95fpdHDcStBtjiu7Jx4V246KZBHJGsnCPwAxyBCaHRRs"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDA5l-s_9KRGiFVIdoIOHF0fWOq9h_nAwM"


# --- SETUP THE RAG CHAIN ---

# 1. Set up the AI model (LLM) - Gemini
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 2. Set up the retriever
# This component will "retrieve" relevant documents from our Pinecone database.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Connect to our existing Pinecone index
vectorstore = PineconeVectorStore(index_name="my-chatbot-index", embedding=embeddings)
# Create a retriever object
retriever = vectorstore.as_retriever()

# 3. Create a Prompt Template
# This tells the AI exactly how to use the information it retrieves.
prompt_template = """
Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Create the RAG Chain
# This chain ties everything together: the retriever, the prompt, and the AI model.
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# --- SETUP THE FASTAPI APP ---

# Create the FastAPI application instance
app = FastAPI(title="Chatbot API")

# Define the structure of the incoming request body
class QueryRequest(BaseModel):
    question: str

# Define the API endpoint that the web team will call
@app.post("/chat")
def chat(query: QueryRequest):
    """
    This endpoint receives a question, processes it through the RAG chain,
    and returns the answer.
    """
    print(f"Received question: {query.question}")
    result = qa_chain({"query": query.question})
    
    print(f"Generated answer: {result['result']}")
    return {"answer": result['result']}

