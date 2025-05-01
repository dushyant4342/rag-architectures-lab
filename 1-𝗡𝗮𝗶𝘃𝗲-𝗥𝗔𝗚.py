"""
➜𝗡𝗮𝗶𝘃𝗲➜𝗥𝗔𝗚 
➜ Retrieve documents, pass them to the LLM, generate an output. 
- Fast to build 
- Fragile when faced with ambiguity, long context, or conflicting information
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1 ▸ load PDF
loader = PyPDFLoader("document_store/DeepSeek.pdf")
docs = loader.load()

# 2 ▸ split
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3 ▸ embed
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(chunks, embedding) #in-memory vector (Data is lost after each run)

#save vectorstore.save_local("vector_db/deepseek_faiss"), reuse vectorstore = FAISS.load_local("vector_db/deepseek_faiss", embedding)

# 4 ▸ LLM
llm = OllamaLLM(model="gemma:7b",   
               # model_kwargs={"num_predict": 300}  # same as max_tokens
                )

# 5 ▸ QA chain
qa = RetrievalQA.from_chain_type(llm=llm,
                                 retriever=vectorstore.as_retriever(search_kwargs={"k":4}),
                                 return_source_documents=True,  # so we can inspect
                                 verbose=False)

# 6 ▸ ask
query = "Explain DeepSeek-R1-Zero and its reinforcement-learning training process."
result = qa.invoke({"query": query})

# 7 ▸ show answer + the retrieved snippets
print("\n🔹 ANSWER:\n", result["result"])
print("\n🔹 RETRIEVED CHUNKS:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n--- chunk {i} ---\n{doc.page_content[:600]} …")