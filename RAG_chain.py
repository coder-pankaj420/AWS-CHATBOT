

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Loading environment variables FIRST, before any library needs them
load_dotenv()


# 1. AI model (LLM)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# 2. Retriever from Pinecone
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="my-chatbot-index", embedding=embeddings)
retriever = vectorstore.as_retriever()

# 3.  Prompt Template
prompt_template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, just say that you don't know. Don't try to make up an answer.

Context: {context}
Question: {question}
Helpful Answer:"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# 4. Final RAG Chain object that the API server will use
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
