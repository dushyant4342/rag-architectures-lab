# Import necessary libraries
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex # Added for summarization capability
)
# Import Settings for global configuration
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama
# Import HuggingFaceEmbedding for local embeddings
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    print("Failed to import HuggingFaceEmbedding from standard path.")
    print("Please ensure you have the necessary package installed:")
    print("pip install llama-index-embeddings-huggingface")
    sys.exit(1) # Exit if import fails

# Agent-related imports
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool

# --- Configuration ---
OLLAMA_MODEL = "gemma:7b" # Or another model available in your Ollama setup
OLLAMA_REQUEST_TIMEOUT = 640.0 # Increased timeout for potentially complex agent reasoning
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DOC_STORE_PATH = "document_store"
TRANSFORMER_PDF = f"{DOC_STORE_PATH}/Attention-Google.pdf"
DEEPSEEK_PDF = f"{DOC_STORE_PATH}/DeepSeek.pdf"
SIMILARITY_TOP_K = 4 # How many results RAG tools should retrieve

# --- Guardrail Configuration ---
MIN_RESPONSE_LENGTH = 20 # Minimum characters for a response to be considered detailed
RELEVANCE_KEYWORDS_THRESHOLD = 1 # Minimum number of query keywords that should appear in the response
FORBIDDEN_PHRASES = ["cannot answer", "don't know", "unable to find", "no information"]

#########################################################################
# 1 ▸ Configure global settings (LLM, Embeddings)
############################################################################
print(f"--- 1. Configuring Global Settings (LLM: {OLLAMA_MODEL}, Embed: {EMBEDDING_MODEL}) ---")
# Instantiate the Ollama LLM
try:
    llm = Ollama(model=OLLAMA_MODEL, request_timeout=OLLAMA_REQUEST_TIMEOUT, temperature=0.1)
    # Perform a simple test call
    llm.complete("Test connection")
    print("Ollama LLM initialized and connection tested.")
except Exception as e:
    print(f"Error initializing Ollama LLM ({OLLAMA_MODEL}): {e}")
    print("Ensure Ollama is running and the model is available.")
    sys.exit(1)

# Instantiate the local embedding model
print(f"Initializing HuggingFace Embedding model ({EMBEDDING_MODEL})...")
try:
    # device="cuda" can be added if GPU is available and configured
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    print("HuggingFace Embedding model initialized.")
except Exception as e:
    print(f"Error initializing HuggingFace Embedding model: {e}")
    print("Ensure 'transformers', 'torch', and 'sentence-transformers' are installed.")
    sys.exit(1)

# Set the global LLM and Embedding Model using Settings
Settings.llm = llm
Settings.embed_model = embed_model
# Optionally configure chunk size
# Settings.chunk_size = 512
print("Global LLM and Embedding Model configured.")

############################################################################
# 2 ▸ Build RAG Query Engines and Tools
############################################################################
print("\n--- 2. Building RAG Query Engines and Tools ---")

def build_vector_query_engine(pdf_path: str, description: str) -> Optional[QueryEngineTool]:
    """Builds a VectorStoreIndex query engine tool from a PDF."""
    print(f"Building RAG tool for: {pdf_path}")
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        print(f"Error: PDF file not found at: {pdf_file.resolve()}")
        return None
    try:
        print(f"Loading data from {pdf_file.resolve()}...")
        docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        print(f"Data loaded. Creating vector index...")
        vector_index = VectorStoreIndex.from_documents(docs)
        print(f"Vector index created. Creating query engine...")
        query_engine = vector_index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K)
        print(f"Query engine created for {pdf_path}.")
        tool = QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=f"rag_{pdf_file.stem.lower().replace('-', '_')}",
                description=description
            )
        )
        print(f"RAG Tool '{tool.metadata.name}' created.")
        return tool
    except Exception as e:
        print(f"Error building RAG tool for {pdf_path}: {e}")
        traceback.print_exc()
        return None

# Build RAG tools for each document
transformer_tool = build_vector_query_engine(
    TRANSFORMER_PDF,
    "Provides information about the 'Attention is All You Need' paper (2017) by Google, focusing on the Transformer architecture, self-attention mechanisms, and model performance."
)
deepseek_tool = build_vector_query_engine(
    DEEPSEEK_PDF,
    "Provides information about the DeepSeek-Coder models, including their architecture, training data, reinforcement learning pipeline (DeepSeek-R1), and performance benchmarks."
)

# Create a list of available tools for the agent
tools = [t for t in [transformer_tool, deepseek_tool] if t is not None]

if not tools:
    print("Error: No RAG tools could be created. Exiting.")
    sys.exit(1)

# --- Optional: Add a Summarization Tool ---
# This requires building a SummaryIndex for each document or combining docs
# For simplicity, we'll create a function tool that summarizes *one* document at a time
# Note: This is less efficient than a dedicated summary index if summarizing often.
def summarize_document(document_name: str) -> str:
    """
    Summarizes the specified document ('transformer' or 'deepseek').
    Args:
        document_name (str): The name of the document to summarize ('transformer' or 'deepseek').
    Returns:
        str: A summary of the document, or an error message.
    """
    print(f"--- Summarization Tool called for: {document_name} ---")
    pdf_path = ""
    if "transformer" in document_name.lower():
        pdf_path = TRANSFORMER_PDF
    elif "deepseek" in document_name.lower():
        pdf_path = DEEPSEEK_PDF
    else:
        return "Error: Invalid document name. Please specify 'transformer' or 'deepseek'."

    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        return f"Error: PDF file not found for {document_name} at {pdf_file.resolve()}"

    try:
        print(f"Loading data for summary from {pdf_file.resolve()}...")
        docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
        print(f"Data loaded. Creating summary index...")
        # You might need to adjust response_mode depending on document length and LLM capability
        summary_index = SummaryIndex.from_documents(docs)
        print(f"Summary index created. Querying for summary...")
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize")
        response = summary_query_engine.query("Provide a concise summary of the key points in this document.")
        print(f"Summary generated for {document_name}.")
        return response.response
    except Exception as e:
        print(f"Error summarizing {document_name}: {e}")
        traceback.print_exc()
        return f"Error generating summary for {document_name}."

summary_tool = FunctionTool.from_defaults(
    fn=summarize_document,
    name="document_summarizer",
    description="Generates a concise summary of the key points from either the 'transformer' paper or the 'deepseek' paper. Specify the document name ('transformer' or 'deepseek') as input."
)
tools.append(summary_tool) # Add the summary tool to the list

print(f"Total tools available for agent: {len(tools)}")
for tool in tools:
    print(f" - Tool: {tool.metadata.name}, Desc: {tool.metadata.description}")


############################################################################
# 3 ▸ Create the ReAct Agent
############################################################################
print("\n--- 3. Creating the ReAct Agent ---")

try:
    # context can be added for more persistent memory or instructions
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True # Set to True to see the agent's reasoning steps
    )
    print("ReAct Agent created successfully.")
except Exception as e:
    print(f"Error creating ReAct Agent: {e}")
    traceback.print_exc()
    sys.exit(1)

############################################################################
# 4 ▸ Define Guardrail Function
############################################################################
print("\n--- 4. Defining Guardrail Function ---")

def apply_guardrails(query: str, response: str) -> (bool, str):
    """
    Checks the agent's response against basic quality criteria.

    Args:
        query (str): The original user query.
        response (str): The agent's final response string.

    Returns:
        tuple[bool, str]: (is_response_good, feedback_message)
                         True if response passes checks, False otherwise.
                         Feedback message explains why it failed or confirms success.
    """
    print("--- Applying Guardrails ---")
    if not response or not response.strip():
        return False, "Guardrail Failed: Response is empty."

    if len(response.strip()) < MIN_RESPONSE_LENGTH:
        return False, f"Guardrail Failed: Response is too short (less than {MIN_RESPONSE_LENGTH} characters)."

    # Check for forbidden phrases indicating failure
    for phrase in FORBIDDEN_PHRASES:
        if phrase in response.lower():
            return False, f"Guardrail Failed: Response contains a problematic phrase ('{phrase}')."

    # Basic relevance check (count keywords from query in response)
    query_keywords = set(word.lower() for word in query.split() if len(word) > 3) # Simple keyword extraction
    response_lower = response.lower()
    matched_keywords = sum(1 for keyword in query_keywords if keyword in response_lower)

    if not query_keywords: # Handle empty or very short queries
         print("Warning: Query too short for effective keyword relevance check.")
    elif matched_keywords < RELEVANCE_KEYWORDS_THRESHOLD:
         return False, f"Guardrail Failed: Response seems irrelevant (only {matched_keywords} query keywords found)."

    print("Guardrail Passed: Response meets basic quality criteria.")
    return True, "Guardrail Passed."

print("Guardrail function defined.")

############################################################################
# 5 ▸ Ask Questions and Apply Guardrails
############################################################################
print("\n--- 5. Running Queries through the Agent ---")

queries = [
    "Explain the multi-head attention mechanism described in the Transformer paper.",
    "What reinforcement learning approach was used for DeepSeek-R1?",
    "Compare the goals of the Transformer paper and the DeepSeek paper.", # Requires using both tools
    "Summarize the DeepSeek paper.", # Should use the summarizer tool
    "What is the capital of France?", # Off-topic, agent might fail or refuse
    "Tell me about self-attention." # Could use transformer tool
]

for q in queries:
    print(f"\n❓ Query: {q}")
    final_response_text = "Agent failed to produce a response." # Default
    try:
        # Use agent.chat for conversational history, agent.query for single turns
        response = agent.query(q)
        final_response_text = response.response if response else "No response object returned."
        print("\n💬 Agent Raw Response:")
        print(final_response_text)

    except Exception as e:
        print(f"🔴 Error processing query with agent: {e}")
        traceback.print_exc()
        final_response_text = f"Error during agent execution: {e}" # Update response text with error

    # Apply Guardrails
    is_good, feedback = apply_guardrails(q, final_response_text)

    print(f"\n✅ Final Answer (Guardrail Status: {'PASS' if is_good else 'FAIL - ' + feedback}):")
    if is_good:
        print(final_response_text)
    else:
        # Provide a fallback or the potentially flawed response with a warning
        print(f"(Warning: Response failed guardrails: {feedback})")
        # print("Fallback: Could not provide a satisfactory answer based on the available documents.")
        print(f"Original Agent Response (for context): {final_response_text}")


print("\n✅ Script finished.")
