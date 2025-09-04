# Gemini-Powered RAG API

A powerful, self-updating Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, and Google's Gemini models. This API allows you to create a searchable knowledge base from your documents and ask questions in natural language.

## Features

-   **Intelligent Q&A:** Uses Google's `gemini-1.5-flash-latest` to provide context-aware answers.
-   **High-Quality Embeddings:** Leverages `text-embedding-004` for accurate document retrieval.
-   **Automatic Indexing:** A background scheduler automatically detects and processes new or modified documents in the `uploads` folder.
-   **RESTful API:** Simple and clean endpoints for uploading documents, querying the system, and deleting documents.
-   **Persistent Vector Store:** Uses ChromaDB to store document embeddings, ensuring data is not lost on restart.
-   **PII Masking:** Automatically detects and masks Personally Identifiable Information in the generated answers for enhanced privacy.
-   **Streaming Responses:** Queries return answers as they are generated for a responsive user experience.
-   **Supported Formats:** Out-of-the-box support for `.pdf` and `.docx` files.

## Architecture

The system is composed of several key components:

1.  **FastAPI Server (`app_google.py`):** Provides the web interface for interacting with the RAG system. It handles file uploads, queries, and document deletion.
2.  **RAG Core (`rag_google.py`):** Contains the core logic for the RAG pipeline, including document loading, chunking, retrieval, and answer generation using LangChain.
3.  **Google Generative AI:** Powers both the document embedding (vector creation) and the final answer generation (LLM).
4.  **ChromaDB:** A vector database used to store and efficiently search through the document embeddings.
5.  **Background Scheduler (`apscheduler`):** A background process that periodically scans the `uploads` directory to keep the knowledge base synchronized.

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   A Google API Key with the "Generative Language API" enabled. You can create one at Google AI Studio.

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

A `requirements.txt` file is provided for easy installation.

```bash
pip install -r requirements.txt
```

### 4. Configure the Application

Open `config_google.json` and replace the placeholder with your actual Google API key.

```json
{
    "google_api_key": "your-google-api-key-here",
    "google_llm": "gemini-1.5-flash-latest",
    "google_embed": "text-embedding-004",
    "chunk_size": 1200,
    "chunk_overlap": 150,
    "k": 4,
    "min_relevance": 0.5,
    "persist_dir": "storage/chroma_google"
}
```

## Running the Application

Once the setup is complete, you can run the FastAPI server using Uvicorn.

```bash
uvicorn app_google:app --host 0.0.0.0 --port 8002 --reload
```

The API will be available at `http://localhost:8002`. You can access the interactive API documentation (Swagger UI) at `http://localhost:8002/docs`.

## API Endpoints

### 1. Upload Documents

Upload one or more documents to be indexed. The files will be saved to the `uploads/` directory and processed automatically.

-   **Endpoint:** `POST /upload`
-   **Request:** `multipart/form-data`

**Example using `curl`:**

```bash
curl -X POST "http://localhost:8002/upload" \
-F "files=@/path/to/your/document1.pdf" \
-F "files=@/path/to/your/document2.docx"
```

### 2. Query the System

Ask a question and receive a streaming answer based on the indexed documents.

-   **Endpoint:** `POST /query`
-   **Request Body:** `application/json`

**Example using `curl`:**

```bash
curl -N -X POST "http://localhost:8002/query" \
-H "Content-Type: application/json" \
-d '{"q": "What is the main conclusion of the project report?"}'
```

### 3. Delete Documents

Remove specified documents and their associated data from the vector store.

-   **Endpoint:** `DELETE /documents`
-   **Request Body:** `application/json`

**Example using `curl`:**

```bash
curl -X DELETE "http://localhost:8002/documents" \
-H "Content-Type: application/json" \
-d '["uploads/document1.pdf", "uploads/document2.docx"]'
```

## Project Structure

```
.
├── uploads/              # Directory for your source documents
├── storage/              # Directory for persistent data
│   ├── chroma_google/    # ChromaDB vector store
│   └── index_state_google.json # Tracks indexed files and modification times
├── app_google.py         # FastAPI application, endpoints, and scheduler
├── rag_google.py         # Core RAG pipeline logic (LangChain)
├── config_google.json    # Configuration file (API keys, model names, etc.)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```