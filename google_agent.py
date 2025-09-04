import os
import json
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from rag_google import retrieve

def create_gemini_agent():
    """
    Creates and configures a ReAct agent powered by the Gemini model.
    This agent is equipped with a tool to search the document vector store.
    """
    # --- 1. Load Configuration ---
    PLACEHOLDER_API_KEY = "your-google-api-key-here"
    cfg_path = "config_google.json"
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            "Configuration file 'config_google.json' not found. "
            "Please run 'app_google.py' first or create the file."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # --- 2. Set API Key and Initialize Models ---
    api_key = cfg.get("google_api_key")
    if not api_key or api_key == PLACEHOLDER_API_KEY:
        raise ValueError(
            "Google API Key not found or is a placeholder in config_google.json. Please create one at https://aistudio.google.com/app/apikey"
        )

    print("Initializing models and vector store...")
    llm = ChatGoogleGenerativeAI(
        model=cfg["google_llm"],
        google_api_key=api_key,
        temperature=0,  # Set to 0 for more deterministic agent behavior
    )

    embeddings = GoogleGenerativeAIEmbeddings(model=cfg["google_embed"], google_api_key=api_key)

    vectorstore = Chroma(
        collection_name="google_rag_collection",
        embedding_function=embeddings,
        persist_directory=cfg["persist_dir"],
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("Initialization complete.")

    # --- 3. Define the RAG Tool ---
    def search_documents(query: str) -> str:
        """
        Searches the vector store for documents relevant to the query
        and returns the formatted context.
        """
        print(f"\n---> Agent using DocumentSearch tool with query: '{query}'")
        results, found = retrieve(
            vectorstore, query, k=cfg["k"], min_relevance=cfg["min_relevance"]
        )
        if not found:
            return "No relevant information found in the documents."
        
        context = "\n\n".join(
            f"[Source: {doc.metadata.get('source', 'N/A')}, Score: {score:.2f}]\n"
            f"{doc.page_content}"
            for doc, score in results
        )
        return context

    tools = [
        Tool(
            name="DocumentSearch",
            func=search_documents,
            description=(
                "Use this tool to find information within the indexed PDF and DOCX files. "
                "It is the primary way to answer user questions about their documents. "
                "The input should be a concise search query."
            ),
        )
    ]

    # --- 4. Create the ReAct Agent ---
    # Pull a prompt template designed for ReAct agents from the LangChain Hub
    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)

    # The AgentExecutor is the runtime for the agent
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor

if __name__ == "__main__":
    # Ensure you have set your GOOGLE_API_KEY in config_google.json

    try:
        llm_agent = create_gemini_agent()
        
        # --- Ask the agent a question ---
        # The agent will decide whether to use the DocumentSearch tool or answer directly.
        question = "What are the main challenges mentioned in the documents regarding AI implementation?"
        
        print(f"\n--- Asking Agent: '{question}' ---")
        response = llm_agent.invoke({"input": question})
        
        print("\n--- Final Answer ---")
        print(response["output"])

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")