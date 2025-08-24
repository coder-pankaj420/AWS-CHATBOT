# RAG Chatbot with Gemini and Pinecone

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses a PDF document as its knowledge base. It is built with LangChain, Google Gemini, and Pinecone, and is served via a FastAPI API.

## Setup Instructions

1.  **Clone the repository:**
    ```
    git clone https://github.com/your-username/aws-rag-chatbot.git
    cd aws-rag-chatbot
    ```

2.  **Create and activate a conda environment:**
    ```
    conda create --name chatbot_env python=3.10
    conda activate chatbot_env
    ```

3.  **Install dependencies:**
    *   This command will install all the necessary packages from the `requirements.txt` file.
    ```
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file:**
    *   In the main project directory, create a file named `.env`.
    *   Add your API keys to this file. Do not use quotes.
    ```
    GOOGLE_API_KEY=your_google_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    ```

5.  **Place your document:**
    *   Place the PDF file you want to use as the knowledge base in the main directory.
    *   Update the `PDF_FILE_PATH` variable in `ingest.py` to match its name.

## How to Run

1.  **Ingest the Data:**
    *   Run this script once to process your PDF and load it into the Pinecone vector database.
    ```
    python ingest.py
    ```

2.  **Run the API Server:**
    *   This command starts the live chatbot server.
    ```
    uvicorn main:app --reload
    ```
    *   The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

*   **Interactive Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
*   **Chat Endpoint:** `POST /chat`
    *   Request Body: `{"question": "Your question here"}`
    *   Response: `{"answer": "The chatbot's answer"}`
