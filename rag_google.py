import os
import json
from typing import List, Tuple, Iterator
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List

def get_similar_queries_from_llm(llm: ChatGoogleGenerativeAI, user_query: str) -> List[str]:
    prompt = f"""You are an intelligent assistant that helps rephrase a user query into semantically similar queries suitable for retrieving information from a document knowledge base.

Given the user query: "{user_query}"

Suggest up to 5 semantically similar query variations in JSON list format, like:
["What is traditional programming?", "Explain procedural programming", "Define traditional software development"]

Return only the JSON list of strings. Do not include any other text or explanation."""

    try:
        print(f"[LLM] Sending prompt for query: {user_query}")
        response = llm.predict(prompt)
        print(f"[LLM] Raw response: '{response}'")
        
        # Clean the response - remove any markdown formatting or extra text
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        print(f"[LLM] Cleaned response: '{response}'")
        
        similar_queries = json.loads(response)
        if isinstance(similar_queries, list) and len(similar_queries) > 0:
            print(f"[LLM] Successfully parsed {len(similar_queries)} similar queries: {similar_queries}")
            return similar_queries
        else:
            print("[LLM] Response is not a valid list or is empty")
            
    except json.JSONDecodeError as e:
        print(f"[Error] JSON decode error: {e}")
        print(f"[Error] Response that failed to parse: '{response}'")
    except Exception as e:
        print(f"[Error] Unexpected error getting similar queries: {e}")

    return []


def load_docs(paths: List[str]) -> List[Document]:
    """Loads documents from the given paths (PDFs and DOCX files)."""
    docs = []
    for path in paths:
        try:
            print(f"[DEBUG] Loading document: {path}")
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                # Clean up any problematic characters
                for doc in loaded_docs:
                    # Replace problematic Unicode characters
                    doc.page_content = doc.page_content.encode('utf-8', errors='ignore').decode('utf-8')
                    # Remove or replace common problematic characters
                    doc.page_content = doc.page_content.replace('\uf0b7', '•')  # bullet point
                    doc.page_content = doc.page_content.replace('\uf020', ' ')   # space
                docs.extend(loaded_docs)
                print(f"[DEBUG] Successfully loaded {len(loaded_docs)} pages from {path}")
            elif path.endswith(".docx"):
                loader = Docx2txtLoader(path)
                loaded_docs = loader.load()
                # Clean up any problematic characters
                for doc in loaded_docs:
                    doc.page_content = doc.page_content.encode('utf-8', errors='ignore').decode('utf-8')
                docs.extend(loaded_docs)
                print(f"[DEBUG] Successfully loaded {len(loaded_docs)} pages from {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue
    print(f"[DEBUG] Total documents loaded: {len(docs)}")
    return docs


def chunk_docs(
    docs: List[Document], chunk_size: int = 1200, chunk_overlap: int = 150
) -> List[Document]:
    """Chunks the documents into smaller pieces."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def add_to_vectorstore(vs: Chroma, docs: List[Document]):
    """Adds new documents to an existing vectorstore and persists the changes."""
    if docs:
        try:
            print(f"[DEBUG] Adding {len(docs)} documents to vectorstore...")
            vs.add_documents(docs)
            print(f"[DEBUG] Successfully added {len(docs)} documents to vectorstore")
        except Exception as e:
            print(f"[ERROR] Failed to add documents to vectorstore: {e}")
            import traceback
            traceback.print_exc()
            raise


def retrieve(
    vs: Chroma, query: str, k: int = 4, min_relevance: float = 0.5
) -> Tuple[List[Tuple[Document, float]], bool]:
    """
    Returns (results, found_flag). Each item is (doc, score).
    Score is a normalized similarity score (higher is better).
    """
    # Debug: Check if vector store has any documents
    try:
        collection_count = vs._collection.count()
        print(f"[DEBUG] Vector store contains {collection_count} documents")
    except Exception as e:
        print(f"[DEBUG] Could not get collection count: {e}")
    
    # similarity_search_with_score returns cosine distance (lower is better).
    # We convert it to a similarity score (higher is better) where 1 is most similar.
    results_with_distance = vs.similarity_search_with_score(query, k=k)
    print(f"[DEBUG] Raw results from similarity search: {len(results_with_distance)} items")
    
    # Debug: Show all results before filtering
    for i, (doc, distance) in enumerate(results_with_distance):
        similarity = 1.0 - distance
        print(f"[DEBUG] Result {i+1}: similarity={similarity:.3f}, distance={distance:.3f}")
        print(f"[DEBUG] Content preview: {doc.page_content[:100]}...")

    # Convert distance to similarity score and filter
    results_with_similarity = []
    for doc, distance in results_with_distance:
        similarity = 1.0 - distance
        if similarity >= min_relevance:
            results_with_similarity.append((doc, similarity))
    
    print(f"[DEBUG] After filtering with min_relevance={min_relevance}: {len(results_with_similarity)} items")
    return results_with_similarity, len(results_with_similarity) > 0


def delete_docs_by_source(vs: Chroma, source_paths: List[str]):
    """Deletes all chunks from the vectorstore that originated from the given source paths."""
    if not source_paths:
        return

    ids_to_delete = []
    for path in source_paths:
        # LangChain loaders store absolute paths in metadata, so we match against that
        abs_path = os.path.abspath(path)
        retrieved = vs.get(where={"source": abs_path})
        if retrieved and retrieved["ids"]:
            ids_to_delete.extend(retrieved["ids"])

    if ids_to_delete:
        vs.delete(ids=ids_to_delete)


def answer_query_with_context(
    query: str,
    retrieved: List[Tuple[Document, float]],
    llm: ChatGoogleGenerativeAI,
) -> Iterator[str]:
    """Generates a streaming answer using the Gemini model and retrieved context."""
    context = "\n\n".join([f"[Source, Score: {score:.2f}] {doc.page_content}" for doc, score in retrieved])

    # prompt_template = ChatPromptTemplate.from_template(
    #     "You are a helpful assistant. Answer the user's question based only on the context provided.\n"
    #     "If the answer is not in the context, state that you cannot answer based on the provided information.\n\n"
    #     "CONTEXT:\n{context}\n\n"
    #     "QUESTION:\n{question}"
    # )

    prompt_template = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the user's question based only on the context provided.\n"
        "If the answer is not in the context, state that you cannot answer based on the provided information.\n"
        "Before answering, detect all Personally Identifiable Information (PII), including names, emails, phone numbers, addresses, and IDs.\n"
        "Mask PII in a granular way:\n"
        " - For names, keep the first letter of the first and last name, replace the rest with * (e.g., 'Schrum HN' → 'S**** H*').\n"
        " - For emails, show first character of username, mask the rest, keep domain (e.g., 'jdoe@gmail.com' → 'j***@gmail.com').\n"
        " - For phone numbers, show last 2 digits, mask the rest (e.g., '123-456-7890' → '***-***-**90').\n"
        " - For IDs, mask all but last 2 characters (e.g., 'AB12345' → '*****45').\n"
        "Maintain formatting and sentence structure while ensuring sensitive info is never fully revealed.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION:\n{question}"
    )

    chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return chain.stream({"context": context, "question": query})