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

def get_context_aware_queries_from_llm(
    llm: ChatGoogleGenerativeAI,
    user_query: str,
    conversation_context: str,
    max_queries: int = 5
) -> List[str]:
    """
    Generate context-aware query variations using conversation history.
    """
    if not conversation_context.strip():
        # Fall back to original function if no conversation context
        return get_similar_queries_from_llm(llm, user_query)

    prompt = f"""You are an intelligent assistant that helps rephrase a user query into semantically similar queries suitable for retrieving information from a document knowledge base.

Given the user query: "{user_query}"

And the conversation context:
{conversation_context}

The conversation context shows what the user has been discussing previously. Use this context to:
1. Understand the domain and topic they're interested in
2. Identify specific entities, concepts, or themes they've mentioned
3. Generate queries that build upon their previous questions and interests
4. Focus on related concepts that would be relevant given the conversation flow

Suggest up to {max_queries} contextually relevant query variations in JSON list format, like:
["What is traditional programming?", "Explain procedural programming", "Define traditional software development"]

Return only the JSON list of strings. Do not include any other text or explanation."""

    try:
        print(f"[LLM] Sending context-aware prompt for query: {user_query}")
        print(f"[LLM] Conversation context length: {len(conversation_context)} characters")
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
            print(f"[LLM] Successfully parsed {len(similar_queries)} context-aware queries: {similar_queries}")
            return similar_queries
        else:
            print("[LLM] Response is not a valid list or is empty, falling back to original query")
            return [user_query]

    except json.JSONDecodeError as e:
        print(f"[Error] JSON decode error: {e}")
        print(f"[Error] Response that failed to parse: '{response}'")
        return [user_query]
    except Exception as e:
        print(f"[Error] Unexpected error getting context-aware queries: {e}")
        return [user_query]

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
