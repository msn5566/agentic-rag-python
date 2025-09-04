# Presentation: Gemini-Powered RAG System

---

## Slide 1: Title Slide

**Title: Gemini-Powered RAG System for Intelligent Document Search**

**Subtitle:** An overview of our new API for Retrieval-Augmented Generation

**Presenter:** [Your Name/Team Name]
**Date:** [Date]

---

## Slide 2: Agenda

-   **The Problem:** How do we find answers within our growing library of documents?
-   **The Solution:** An overview of Retrieval-Augmented Generation (RAG).
-   **System Architecture:** A high-level look at the components.
-   **Core Engine (`rag_google.py`):** How the "magic" happens.
-   **API Layer (`app_google.py`):** How we interact with the system.
-   **Key Feature: Automated Indexing:** Keeping our knowledge base fresh.
-   **Live Demo / Workflow:** Putting it all together.
-   **Next Steps & Q&A**

---

## Slide 3: The Problem: Unstructured Data Overload

-   Our organization has a large and growing number of documents (`.pdf`, `.docx`, etc.).
-   Finding specific information is time-consuming and inefficient.
-   Simple keyword searches often miss context and fail to provide direct answers.

**Question:** How can we enable users to ask natural language questions and get direct, accurate answers from our documents?

---

## Slide 4: The Solution: Retrieval-Augmented Generation (RAG)

We've built a system that combines two powerful AI techniques:

1.  **Retrieval:** A specialized database (ChromaDB) finds the most relevant text snippets from our documents based on the user's query.
2.  **Generation:** A powerful Large Language Model (Google's Gemini 1.5 Flash) uses these snippets as context to generate a concise, human-like answer.

**It's like an open-book exam for an AI—it finds the facts first, then formulates the answer.**

---

## Slide 5: System Architecture

```
                               +----------------------+
                               | Google Gemini LLM    |
                               +----------+-----------+
                                          ^
                                          | 5. Generate Answer
                                          |
+------+   1. API Request   +-----------------+   4. Send Query + Context   +----------------+
| User |------------------->|  FastAPI App    |---------------------------->| RAG Core Engine|
+------+  (Upload, Query)  | (app_google.py) |                             | (rag_google.py)|
         <------------------|                 |<----------------------------|                |
          6. Stream Answer |                 |   3. Retrieve Context       +-------+--------+
                           +--------+--------+                                     |
                                    | 2. Index / Store                            |
                                    v                                             v
                           +--------+--------+                           +--------+--------+
                           |  Uploads Folder |                           | Chroma VectorDB |
                           +-----------------+                           +-----------------+
                                    ^
                                    |
                           +--------+--------+
                           |  Auto-Scheduler |
                           +-----------------+
```

---

## Slide 6: The Core Engine (`rag_google.py`)

This module contains the fundamental logic for our RAG pipeline.

1.  **Load & Chunk (`load_docs`, `chunk_docs`):**
    -   Accepts `.pdf` and `.docx` files.
    -   Splits large documents into smaller, overlapping text "chunks" (`~1200` characters). This is crucial for finding precise context.

2.  **Vectorize & Store (`add_to_vectorstore`):**
    -   Uses Google's `text-embedding-004` model to convert each text chunk into a numerical vector (an "embedding").
    -   Stores these vectors in our persistent ChromaDB vector store.

3.  **Retrieve (`retrieve`):**
    -   When a query comes in, it's also converted to a vector.
    -   ChromaDB performs a "cosine similarity" search to find the text chunks with vectors most similar to the query's vector.
    -   Filters out results below a relevance threshold (`0.5`) to ensure quality.

---

## Slide 7: Core Engine: Smart Generation & Security

Our generation process is more than just asking a question.

**`answer_query_with_context` function:**

-   **Context-Augmented Prompting:** The retrieved text chunks are dynamically inserted into a carefully crafted prompt. The LLM is instructed to **answer based only on this context**.
-   **PII Masking:** Before generating the final answer, the prompt instructs the LLM to identify and mask Personally Identifiable Information (PII) in a granular way.

**Example of PII Masking:**
-   **Name:** `John Smith` -> `J*** S****`
-   **Email:** `test@example.com` -> `t***@example.com`
-   **Phone:** `123-456-7890` -> `***-***-**90`

This is a critical security and privacy feature built directly into the generation step.

---

## Slide 8: The API Layer (`app_google.py`)

This FastAPI application serves as the user-friendly interface to our RAG engine.

-   **`POST /upload`**: Accepts one or more files, saves them, and triggers indexing.
-   **`POST /query`**: Accepts a JSON body `{"q": "your query"}` and streams the LLM's response back in real-time.
-   **`DELETE /documents`**: Accepts a list of file paths to remove from the vector store.

---

## Slide 9: Key Feature: Automated Indexing

The system is designed to be low-maintenance and always up-to-date.

1.  **Startup Scan:** When the API server starts, it immediately scans the `uploads/` folder for any new or modified files and indexes them.
2.  **Background Scheduler (`apscheduler`):**
    -   A background job runs automatically every **5 minutes**.
    -   It re-scans the `uploads/` folder.
    -   **New Files:** If a new file is found, it's added to the vector store.
    -   **Modified Files:** If a file has been changed, the old version is deleted and the new version is indexed.

**Benefit:** We can simply add, update, or replace files in the `uploads` folder, and the RAG system will automatically sync its knowledge base.

---

## Slide 10: Next Steps & Discussion

-   **Potential Enhancements:**
    -   Building a simple web-based UI for uploads and queries.
    -   Adding support for more file types (e.g., `.txt`, `.pptx`).
    -   Implementing user authentication and authorization.
-   **Open for Discussion**

---

## Slide 11: Q&A

**Thank You!**