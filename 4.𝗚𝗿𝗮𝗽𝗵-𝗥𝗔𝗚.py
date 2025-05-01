"""
https://www.dailydoseofds.com/a-crash-course-on-building-rag-systems-part-7-with-implementation/#read-the-documents

Below is a bare-bones “Graph RAG” starter that:
extracts subject-predicate-object triples from each PDF chunk,
writes them into a local Neo4j graph,
runs a Cypher-powered QA chain with your gemma:7b Ollama model.

# One-time
brew install neo4j                # or docker run neo4j:latest
neo4j start                       # defaults → bolt://localhost:7687, neo4j / password
pip install -U langchain langchain-community langchain-ollama \
                neo4j spacy transformers sentencepiece
python -m spacy download en_core_web_sm
ollama serve
ollama pull gemma:7b

Use the graph inspector (http://localhost:7474) to visually trace why an answer was produced — that’s the “explainable AI” edge Graph RAG brings.
Feel free to swap the simple spaCy extractor for an LLM-powered one (e.g., gemma:7b + prompt “output triples”), or connect to a managed Neo4j/Aura instance for production.
graph_rag.py  ::  Graph-based RAG with Neo4j + gemma:7b
------------------------------------------------------
Creates a mini knowledge-graph from DeepSeek.pdf then answers a question
via GraphCypherQAChain.
"""

PASSWORD = "sting-invite-adrian-charm-random-7964"



# graph_rag_attention_final.py  – tested with:
#   langchain-community 0.2.1
#   langchain-neo4j      0.0.8
#   neo4j                5.x
#   spacy                3.7.x
#   ollama               latest (gemma:7b)



# graph_rag_attention_working.py – one‑file Graph‑RAG that runs on latest langchain‑neo4j (0.4.x)
# -------------------------------------------------------------------------------------------
# What it does
#   • Loads Attention‑Google.pdf
#   • Extracts simple (subject, verb, object) triples with spaCy
#   • Stores them in local Neo4j WITHOUT requiring the APOC plugin
#   • Lets you query with natural language via GraphCypherQAChain + Gemma‑7B (Ollama)
#   • Prints progress logs so you see every step
#   • Safely falls back if the LLM cannot generate Cypher
#
# Prereqs (run once in your rag‑env venv)
#   pip install -U "langchain~=0.2" langchain-community langchain-neo4j==0.4.* spacy neo4j
#   python -m spacy download en_core_web_sm
#   ollama serve   &&   ollama pull gemma:7b
#   # Neo4j must be running on bolt://localhost:7687 (brew or Desktop is fine)

from pathlib import Path
import re, time, textwrap, spacy
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_ollama import OllamaLLM

# --- IMPORT THE GRAPH STORE --------------------------------------------------
# langchain‑neo4j 0.4.x relocated the class → graph_stores.neo4j_graph_store
try:
    from langchain_neo4j.graph_stores.neo4j_graph_store import Neo4jGraphStore
except ImportError:  # very old langchain‑neo4j (<0.0.8)
    from langchain_neo4j import Neo4jGraph as Neo4jGraphStore

# ---- CONFIG -----------------------------------------------------------------
PDF_PATH   = "document_store/Attention-Google.pdf"
BOLT_URL   = "bolt://localhost:7687"
USERNAME   = "neo4j"
#PASSWORD   = "your_neo4j_password"   # ← change
CHUNK_SIZE = 700
CHUNK_OVLP = 150

# ── 1. spaCy tiny triple extractor -------------------------------------------
print("⏳ loading spaCy …")
nlp = spacy.load("en_core_web_sm")

def extract_triples(text: str, limit: int = 8):
    triples = []
    for sent in nlp(text).sents:
        subj = [t for t in sent if t.dep_ == "nsubj"]
        obj  = [t for t in sent if t.dep_ in {"dobj", "pobj"}]
        if subj and obj:
            triples.append((subj[0].text, sent.root.lemma_, obj[0].text))
        if len(triples) >= limit:
            break
    return triples

# ── 2. Connect to Neo4j via GraphStore (NO APOC) -----------------------------
print("🔌 connecting Neo4j …")
store = Neo4jGraphStore(
    url=BOLT_URL,
    username=USERNAME,
    password=PASSWORD,
   # include_schema=True,   # ← disables apoc.meta.data()
)

# ── 3. Ingest PDF → triples ---------------------------------------------------
chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                        chunk_overlap=CHUNK_OVLP).split_documents(
    PyPDFLoader(PDF_PATH).load())
print(f"📄 {len(chunks)} chunks in {Path(PDF_PATH).name}")

n_triples = 0
for i, ch in enumerate(chunks, 1):
    t = extract_triples(ch.page_content)
    if not t:
        continue
    print(f"  • chunk {i:02d} → {len(t)} triples")
    n_triples += len(t)
    for s, p, o in t:
        rel = re.sub(r"\W+", "_", p).upper()
        if not rel[0].isalpha():
            rel = "REL_" + rel
        store.query(
            f"""
            MERGE (a:Entity {{name:$s}})
            MERGE (b:Entity {{name:$o}})
            MERGE (a)-[:{rel}]->(b)
            """,
            params={"s": s, "o": o},
        )
print(f"✅ inserted {n_triples} triples\n")

print("📊 schema preview → first 15 relationship types")
rows = store.query("MATCH ()-[r]->() RETURN DISTINCT type(r) AS rel LIMIT 15")
print(", ".join(r["rel"] for r in rows))

# ── 4. Graph‑aware QA chain ---------------------------------------------------
llm = OllamaLLM(model="gemma:7b", model_kwargs={"num_predict": 200})
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=store,
    allow_dangerous_requests=True,  # required flag
    top_k=12,
    verbose=True,
)

def ask(question: str) -> str:
    """Ask safely – if LLM fails to produce Cypher, return a warning."""
    cypher = chain.generate_cypher(question)
    if not cypher.upper().lstrip().startswith(("MATCH", "CALL", "CREATE")):
        return "⚠️ No matching triples in the graph."
    ctx = store.query(cypher)[: chain.top_k]
    return chain.answer_question(question, ctx)

if __name__ == "__main__":
    q = "Why is attention important in the Transformer paper?"
    t0 = time.time()
    print(f"\n❓ {q}\n")
    print("🟢", ask(q))
    print(f"\n⏱ {time.time()-t0:.1f}s")

# How to inspect the graph visually
# 1. open http://localhost:7474/browser/preview/  → log in with same creds
# 2. run:  MATCH (a)-[r]->(b) RETURN a,r,b LIMIT 100  → switch to graph view
