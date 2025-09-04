import os
from functools import lru_cache
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document

def load_docs(paths: List[str]) -> List[Document]:
    docs: List[Document] = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(p)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(p)
        else:
            # Fallback: treat as plain text
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(p, encoding="utf-8")
        docs.extend(loader.load())
    return docs

def chunk_docs(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)

def add_to_vectorstore(
    vs: Chroma,
    docs: List[Document]
):
    """Adds new documents to an existing vectorstore and persists the changes."""
    if docs:
        vs.add_documents(docs)
        vs.persist()

def retrieve(
    vs: Chroma,
    query: str,
    k: int = 4,
    min_relevance: float = 0.7
) -> Tuple[List[Tuple[Document, float]], bool]:
    """Returns (results, found_flag). Each item is (doc, score). Score is cosine distance (lower is better)."""
    # Use the built-in score_threshold argument, which is more robust.
    # This delegates the filtering to the vector store, which knows how to handle its native scores.
    results = vs.similarity_search_with_score(query, k=k)

    # Manually filter by relevance score
    filtered_results = [res for res in results if res[1] <= min_relevance]

    return filtered_results, len(filtered_results) > 0

def delete_docs_by_source(
    vs: Chroma,
    source_paths: List[str]
):
    """Deletes all chunks from the vectorstore that originated from the given source paths."""
    if not source_paths:
        return

    ids_to_delete = []
    for path in source_paths:
        # LangChain loaders store absolute paths in metadata, so we match against that
        abs_path = os.path.abspath(path)
        retrieved = vs.get(where={"source": abs_path})
        if retrieved and retrieved['ids']:
            ids_to_delete.extend(retrieved['ids'])

    if ids_to_delete:
        vs.delete(ids=ids_to_delete)

@lru_cache(maxsize=4)
def get_llm(model_name: str) -> OllamaLLM:
    """Use a cache to avoid re-initializing the LLM on every call."""
    return OllamaLLM(
        model=model_name,
        temperature=0.7,  # More focused responses
        timeout=30,  # Fail fast if slow
        num_ctx=2048,  # Smaller context window
        num_thread=4  # Optimize CPU usage
    )

def answer_query_with_context(
    query: str,
    retrieved: List[Tuple[Document, float]],
    model_name: str
) -> str:
    llm = get_llm(model_name)
    context = "\n\n".join([f"[Score {score:.2f}] {doc.page_content}" for doc, score in retrieved])
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant answering strictly from the provided context.\n"
        "If the answer is not in the context, state that you cannot answer based on the provided information.\n\n"
        "Question:\n"
        "{question}\n\n"
        "Context:\n"
        "{context}\n\n"
        "Answer concisely:"
    )
    
    # Enable streaming for faster response display
    for chunk in llm.stream(prompt.format(question=query, context=context)):
        yield chunk
