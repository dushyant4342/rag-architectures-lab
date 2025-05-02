"""
Autonomous routing   one query → the router reads the tool descriptions, chooses the best retriever, and forwards the call.
Extensible   add more tools (SQL, Graph, APIs) and the router automatically reasons over them.
Local LLM   uses your Ollama Gemma 7B as both the router brain and the response generator.

| Phase | What happens |
|-------|--------------|
| **Build** | Each PDF → its own **VectorStoreIndex** (separate embeddings & faiss files). |
| **Tool wrap** | `QueryEngineTool` gives the router a *name* + *description* + *query fn*. |
| **Routing** | The router LLM (Gemma-7B) receives the user query & tool descriptions, picks *one* tool, and forwards the query. |
| **Answering** | The selected index retrieves top-k chunks, the same Gemma-7B composes the final answer, and LlamaIndex returns it. |
| **Extending** | Drop in more tools (SQL, Graph, web-search)  the router prompt stays the same. |

"""

#https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/



# Import necessary libraries
from pathlib import Path
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)
# Import Settings for global configuration
from llama_index.core.settings import Settings
from llama_index.llms.ollama import Ollama

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    print("Failed to import HuggingFaceEmbedding from standard path.")
    print("Please ensure you have the necessary package installed:")
    print("pip install llama-index-embeddings-huggingface")
    exit() # Exit if import fails

from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine


#########################################################################
# 1 ▸ Configure global settings with Ollama Gemma-7B and Local Embeddings
############################################################################
# Instantiate the Ollama LLM
# Increased timeout for potentially slower local models
llm = Ollama(model="gemma:7b", request_timeout=6000.0, temperature=0.1, num_predict=256)

# Instantiate the local embedding model
# Using 'BAAI/bge-small-en-v1.5' as a good default local model
# Ensure necessary base libraries are installed: pip install transformers torch sentence-transformers
# Also potentially: pip install llama-index-embeddings-huggingface
print("Initializing HuggingFace Embedding model...")
try:
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("HuggingFace Embedding model initialized.")
except Exception as e:
    print(f"Error initializing HuggingFace Embedding model: {e}")
    print("Ensure 'transformers', 'torch', and 'sentence-transformers' are installed.")
    exit()


# Set the global LLM and Embedding Model using Settings
Settings.llm = llm
Settings.embed_model = embed_model # Explicitly set the local embedding model

# Optionally configure chunk size
# Settings.chunk_size = 512

############################################################################
# 2 ▸ Build two independent vector indexes
############################################################################
def build_index(pdf_path: str):
    """
    Builds a VectorStoreIndex from a PDF file using global settings.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        VectorStoreIndex: The created index.
    """
    print(f"Building index for: {pdf_path}")
    # Load documents from the specified PDF file
    # Ensure the file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file not found at: {pdf_file.resolve()}")
    print(f"Loading data from {pdf_file.resolve()}...")
    docs = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    print(f"Data loaded. Creating index...")
    # Create the index from documents. It will use the global Settings (LLM, embed_model).
    index = VectorStoreIndex.from_documents(docs)
    print(f"Index built successfully for: {pdf_path}")
    return index

# Define file paths (adjust if necessary)
transformer_pdf_path = "document_store/Attention-Google.pdf"
deepseek_pdf_path = "document_store/DeepSeek.pdf"

try:
    index_transformer = build_index(transformer_pdf_path)
    index_deepseek    = build_index(deepseek_pdf_path)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please ensure the PDF files exist in the 'document_store' directory relative to the script.")
    exit() # Exit if files are missing
except Exception as e:
    print(f"An error occurred during index building: {e}")
    # Print detailed traceback for debugging if needed
    import traceback
    traceback.print_exc()
    exit()


############################################################################
# 3 ▸ Wrap each index as a “tool” with a natural-language description
############################################################################
print("Creating query engine tools...")
# Create a query engine tool for the Transformer paper index
tool_transformer = QueryEngineTool.from_defaults(
    # Create a query engine from the index (uses global Settings)
    query_engine=index_transformer.as_query_engine(similarity_top_k=4),
    name="transformer_paper_engine",
    description="Good for questions about the 2017 Google paper “Attention is All You Need” and the Transformer architecture."
)

# Create a query engine tool for the DeepSeek paper index
tool_deepseek = QueryEngineTool.from_defaults(
    # Create a query engine from the index (uses global Settings)
    query_engine=index_deepseek.as_query_engine(similarity_top_k=4),
    name="deepseek_report_engine",
    description="Covers DeepSeek-R1 and reinforcement-learning research details."
)

# List of tools for the router
tools = [tool_transformer, tool_deepseek]
print("Query engine tools created.")

############################################################################
# 4 ▸ Create the RouterQueryEngine (agent)
############################################################################
print("Creating Router Query Engine...")
# Create the router query engine which selects the appropriate tool
# --- Removed system_prompt/router_system_prompt parameter ---
# Let's rely on the default selector prompt for now.
# The selector should automatically use the LLM from Settings.
try:
    router_engine = RouterQueryEngine.from_defaults(
        query_engine_tools=tools,
        verbose=True # Uncomment for more detailed logging from the router
    )
    print("Router Query Engine created.")
except Exception as e:
    print(f"Error creating RouterQueryEngine: {e}")
    import traceback
    traceback.print_exc()
    exit()


############################################################################
# 5 ▸ Ask some questions
############################################################################
queries = [
    "Explain why attention is important in the Transformer paper.",
    "What reinforcement-learning pipeline is used in DeepSeek-R1?",
    "Summarize the main contribution of the 'Attention is All You Need' paper.",
    "Describe the architecture of DeepSeek-R1."
]

# Iterate through queries and get answers from the router engine
for q in queries:
    print(f"\n❓ Query: {q}")
    try:
        answer = router_engine.query(q)
        # Accessing the response string
        print("🟢 Answer:", answer.response)
        # Optionally print metadata like selected source nodes
        # print("Metadata:", answer.metadata) # Often contains selected tool info
    except Exception as e:
        print(f"🔴 Error processing query: {q}")
        print(f"   Error: {e}")
        # Print detailed traceback for debugging if needed
        import traceback
        traceback.print_exc()


print("\n✅ Script finished.")
