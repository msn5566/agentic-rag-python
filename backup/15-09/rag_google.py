import os
from typing import List, Tuple, Iterator
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

def load_docs(paths: List[str]) -> List[Document]:
    """Loads documents from the given paths (PDFs and DOCX files)."""
    docs = []
    for path in paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
            docs.extend(loader.load())
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
        vs.add_documents(docs)


def retrieve(
    vs: Chroma, query: str, k: int = 4, min_relevance: float = 0.5
) -> Tuple[List[Tuple[Document, float]], bool]:
    """
    Returns (results, found_flag). Each item is (doc, score).
    Score is a normalized similarity score (higher is better).
    """
    # similarity_search_with_score returns cosine distance (lower is better).
    # We convert it to a similarity score (higher is better) where 1 is most similar.
    results_with_distance = vs.similarity_search_with_score(query, k=k)

    # Convert distance to similarity score and filter
    results_with_similarity = []
    for doc, distance in results_with_distance:
        similarity = 1.0 - distance
        if similarity >= min_relevance:
            results_with_similarity.append((doc, similarity))

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