

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Import the RAG model
from RAG_chain import qa_chain

# Loading environment variables FIRST
load_dotenv()

# Creating the main FastAPI application instance
app = FastAPI(title="AWS CHATBOT")


class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(query: QueryRequest):
    """
    Receives a question, passes it to the imported RAG chain,
    and returns the answer.
    """
    print(f"Received question: {query.question}")
    result = qa_chain.invoke({"query": query.question})
    
    print(f"Generated answer: {result['result']}")
    return {"answer": result['result']}


@app.get("/")
def read_root():
    return {"status": "API is running"}
