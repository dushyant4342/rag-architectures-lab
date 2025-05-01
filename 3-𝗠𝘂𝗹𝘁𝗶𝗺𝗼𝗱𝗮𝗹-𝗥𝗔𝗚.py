"""
multimodal_rag.py  –  text + image RAG with Ollama llama3.2-vision
------------------------------------------------------------------
Folder layout:
  document_store/
      DeepSeek.pdf
      images/
          xray_1.jpg
          contract_page.png
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document   # ← add this import


################################################################################
# 1 ▸ build / load vector DB
################################################################################

PDF_PATH = "document_store/DeepSeek.pdf"
IMG_DIR  = Path("document_store/images")
DB_PATH  = "vector_db/mm_faiss"
ENC = OpenCLIPEmbeddings()          # ViT-B/32  → 512-d


TXT_EMB  = HuggingFaceEmbeddings(
              model_name="sentence-transformers/all-mpnet-base-v2")

#IMG_EMB  = OpenCLIPEmbeddings()                       # ViT-B/32, LAION-400M
# txt_store	all-mpnet-base-v2	768
# img_store	OpenCLIP (ViT-B/32)	512



# create two mini-stores and merge
if not Path(DB_PATH).exists():
    print("‣ Building multimodal FAISS index …")

    # ---- text chunks --------------------------------------------------------

    # ---- build text store with CLIP instead of mpnet
    pdf_chunks = TokenTextSplitter(chunk_size=256, chunk_overlap=64) \
                .split_documents(PyPDFLoader(PDF_PATH).load())
    txt_store  = FAISS.from_documents(pdf_chunks, ENC)

    # ---- build image store (same ENC)
    img_docs = []
    captioner = OllamaLLM(model="llama3.2-vision")
    for img_path in IMG_DIR.glob("*.[jp][pn]g"):
        cap = captioner.invoke(
            f"Describe the following image in two sentences:\n<image:{img_path}>")
        img_docs.append(Document(page_content=cap,
                                metadata={"src": str(img_path), "type": "image"}))

    img_store = FAISS.from_documents(img_docs, ENC)


    # ---- now they’re 512-d each, so merge works
    txt_store.merge_from(img_store)
    txt_store.save_local(DB_PATH)
    mixed_vectors = txt_store

    # pdf_chunks = TokenTextSplitter(
    #                 chunk_size=256, chunk_overlap=64
    #              ).split_documents(PyPDFLoader(PDF_PATH).load())
    # txt_store  = FAISS.from_documents(pdf_chunks, TXT_EMB)

    # # ---- image chunks -------------------------------------------------------
    # img_docs = []
    # captioner = OllamaLLM(model="llama3.2-vision")

    # for img_path in IMG_DIR.glob("*.[jp][pn]g"):
    #     cap = captioner.invoke(
    #         f"Describe the following image in one sentence:\n<image:{img_path}>")
    #     img_docs.append(
    #         Document(
    #             page_content=cap,
    #             metadata={"src": str(img_path), "type": "image"}
    #         )
    #     )
    # img_store = FAISS.from_documents(img_docs, IMG_EMB)
    # # ---- merge --------------------------------------------------------------
    # txt_store.merge_from(img_store)
    # txt_store.save_local(DB_PATH)
    # mixed_vectors = txt_store

else:
    mixed_vectors = FAISS.load_local(DB_PATH, TXT_EMB)   # loads all vectors

################################################################################
# 2 ▸ Retrieval + answer generation
################################################################################

llm    = OllamaLLM(model="llama3.2-vision")
prompt = PromptTemplate.from_template(
    "Answer using ONLY the context (text or image captions).\n\nContext:\n{ctx}\n\nQ: {q}\nA:"
)
chain  = prompt | llm | StrOutputParser()

def multimodal_qa(question, k=6):
    ctx_docs = mixed_vectors.similarity_search(question, k=k)
    ctx      = "\n\n".join(d.page_content for d in ctx_docs)
    answer   = chain.invoke({"ctx": ctx, "q": question})
    return answer, ctx_docs

if __name__ == "__main__":
    q  = "Summarise the Transformers architecture and explain what 'attention is all you need' means."
    ans, sources = multimodal_qa(q)
    print("\n🔹 ANSWER:\n", ans)
    print("\n🔹 SOURCES:")
    for d in sources:
        print("-", d.metadata.get("src", "pdf-chunk"), "…", d.page_content[:80])
