# main.py

# --- FIRST: Load Environment Variables ---
import os
from dotenv import load_dotenv

# This is the most important line. It MUST run before other imports.
load_dotenv()


# --- SECOND: Import all other libraries ---
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


# --- SETUP THE RAG CHAIN ---

# 1. Set up the AI model (LLM) - Gemini
# Now, when this line runs, the GOOGLE_API_KEY is already loaded and available.
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 2. Set up the retriever
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="my-chatbot-index", embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3. Create a Prompt Template
prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.

Context: {context}

Question: {question}

Helpful Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Create the RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)


# --- SETUP THE FASTAPI APP ---
app = FastAPI(title="Chatbot API")

class QueryRequest(BaseModel):
    question: str

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
