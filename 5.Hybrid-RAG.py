# hybrid_rag.py  ────────────────────────────────────────────────────────────
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

DOCS_DIR   = "document_store"
PDF_FILES  = [str(p) for p in Path(DOCS_DIR).glob("*.pdf")]

############################################################################
# 1 ▸ Load & chunk docs
############################################################################
loader   = PyPDFLoader(PDF_FILES)
docs_all = []
for pdf in PDF_FILES:
    docs_all += RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=120
    ).split_documents(PyPDFLoader(pdf).load())

############################################################################
# 2 ▸ Build vector & keyword retrievers
############################################################################
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
faiss_store = FAISS.from_documents(docs_all, emb)
vec_retriever = faiss_store.as_retriever(search_kwargs={"k":6})

bm25_retriever = BM25Retriever.from_documents(docs_all)
bm25_retriever.k = 6

############################################################################
# 3 ▸ (Optional) add a graph retriever
############################################################################
# from langchain_neo4j.graph_store import Neo4jGraphStore
# graph_store = Neo4jGraphStore(url="bolt://localhost:7687",
#                               username="neo4j", password="pw",
#                               include_schema=False)
# def graph_retrieve(query: str, k=6):
#     cypher = f"CALL db.index.fulltext.queryNodes('entityIndex', '{query}') YIELD node RETURN node LIMIT {k}"
#     return graph_store.query(cypher)
# class GraphRetriever(BaseRetriever):
#     def get_relevant_documents(self, query): return graph_retrieve(query)
# graph_retriever = GraphRetriever()

############################################################################
# 4 ▸ Ensemble (vector + keyword [+ graph])
############################################################################
retriever = EnsembleRetriever(
    retrievers=[vec_retriever, bm25_retriever],   # add graph_retriever here
    weights   =[0.6, 0.4]                        # adjust as needed
)

############################################################################
# 5 ▸ LLM + QA chain
############################################################################
llm  = OllamaLLM(model="gemma:7b", model_kwargs={"num_predict":256})
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",          # simple prompt
    return_source_documents=True
)

############################################################################
# 6 ▸ Ask a question
############################################################################
query = "Why is attention crucial in the Transformer architecture?"
result = chain({"query": query})
print("\nAnswer:\n", result["result"])
print("\nSources:")
for doc in result["source_documents"]:
    print("-", doc.metadata.get("source"), "…", doc.page_content[:80], "…")
