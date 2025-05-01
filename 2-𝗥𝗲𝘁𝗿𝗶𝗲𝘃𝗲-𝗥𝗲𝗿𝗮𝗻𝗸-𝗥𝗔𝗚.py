"""
Retrieve + Cross-Encoder-Rerank RAG (Ollama + gemma:7b)

What changed vs Naive
search 15 ➜ rerank (cross-encoder) ➜ keep 4 : sharpens relevance
prompt fed only those top-4 chunks → cleaner, lower hallucination risk

Retrive&ReRankRAG
➜ Adds reranking to prioritize the most relevant information before generation. 
- Improves accuracy and grounding 
- Reduces risk of hallucinations

🔁 Why use two different models?
Step	    Model Type	                            Why This Model?
Embedding	all-mpnet-base-v2 (bi-encoder)	        Fast & scalable for vector search; good semantic similarity; precomputes everything
Reranking	ms-marco-MiniLM-L-6-v2 (cross-encoder)	deep pairwise re-reading → slower but much sharper ranking(More accurate, because it compares [query + doc] together at inference time)
"""

from pathlib import Path
from sentence_transformers import CrossEncoder

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

PDF_PATH   = "document_store/DeepSeek.pdf"
VDB_PATH   = "vector_db/deepseek_faiss"
EMB_MODEL  = "sentence-transformers/all-mpnet-base-v2"
RERANKER   = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 0 ▸ vector DB (load or build once)
embedding = HuggingFaceEmbeddings(model_name=EMB_MODEL)
if Path(VDB_PATH).exists():
    vectorstore = FAISS.load_local(VDB_PATH, embedding)
else:
    docs = PyPDFLoader(PDF_PATH).load()
    chunks = RecursiveCharacterTextSplitter(
                chunk_size=800, chunk_overlap=200).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(VDB_PATH)
    print("VECTOR DB CREATED")

# 1 ▸ cross-encoder reranker
reranker = CrossEncoder(RERANKER)

def retrieve_and_rerank(query, k_search=15, k_final=4):
    # coarse retrieval
    docs = vectorstore.similarity_search(query, k=k_search)    ########similarity_search
    # score pairs
    pairs  = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)    #ms-marco-MiniLM-L-6-v2 → slow, smart scoring of best 4 chunks
    # top-k reranked
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [d for d, _ in ranked[:k_final]]

# 2 ▸ LLM
llm = OllamaLLM(model="gemma:7b")

prompt = PromptTemplate.from_template(
    "Use ONLY the context to answer concisely.\n\nContext:\n{ctx}\n\nQ: {q}\nA:")

print(f'prompt:{prompt}')
qa_chain = LLMChain(llm=llm, prompt=prompt)

def answer(query):
    top_docs = retrieve_and_rerank(query)
    ctx = "\n\n".join(d.page_content for d in top_docs)
    return qa_chain.invoke({"ctx": ctx, "q": query}), top_docs

# 3 ▸ demo
if __name__ == "__main__":
    q = "Explain DeepSeek-R1-Zero and its reinforcement-learning training process."
    ans, docs = answer(q)
    print("\n🔹 ANSWER:\n", ans["text"])
    print("\n🔹 RERANKED CHUNKS:")
    for i, d in enumerate(docs, 1):
        print(f"\n--- chunk {i} ---\n{d.page_content[:600]} …")
