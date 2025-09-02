

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader  
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec




load_dotenv() # Load variables from .env

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")




PDF_FILE_PATH = "AWS Description.pdf" 

PINECONE_INDEX_NAME = "my-chatbot-index"

def main():
    """
    This is the main function that runs our ingestion process.
    """
    print("Starting data ingestion...")

    # Loading the PDF document
    print(f"Loading PDF: {PDF_FILE_PATH}...")
    loader = PyPDFLoader(PDF_FILE_PATH)
    documents = loader.load()
    if not documents:
        print("Error: Could not load any documents from the PDF.")
        return
    print(f"Loaded {len(documents)} pages from the PDF.")

    # Spliting the document into smaller chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split into {len(chunked_documents)} chunks.")

    # embeddings
    print("Creating text embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- NEW CODE BLOCK TO CREATE THE INDEX ---
    print("Connecting to Pinecone...")
    pc = Pinecone() # The API key is now read automatically from the environment variable

    # Check if the index already exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        # If not, create a new index
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # This must match the dimension of your embeddings model
            metric='cosine', # This is a standard metric for similarity search
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
            
            
        )
        print("Index created successfully.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists. Proceeding...")
    # --- END OF NEW CODE BLOCK ---


    # Uploading the chunks to the (now existing) Pinecone index
    print("Uploading data to Pinecone index...")
    PineconeVectorStore.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )

    print("✅ Ingestion complete!")
    print(f"Your data has been uploaded to the '{PINECONE_INDEX_NAME}' index in Pinecone.")


# This makes the script runnable from the command line
if __name__ == "__main__":
    main()
