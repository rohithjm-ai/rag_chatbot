from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from pathlib import Path

folder_path = Path(r"C:\rag_chatbot\data")
pdf_files = folder_path.glob("*.pdf")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = []
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

for pdf in pdf_files:
    print(f"LOADING : {pdf.name}")
    loader = PyPDFLoader(str(pdf))
    pdf_documents = loader.load()
    print(f"Size of {pdf.name} : {len(pdf_documents)}")
    documents.extend(pdf_documents)
chunks = splitter.split_documents(documents)
print(f"Total Document : {len(documents)}")
print(f"Toal Chunks : {len(chunks)}")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
while True:
    query = input("You : ")
    if query.lower() == "exit":
        break
    llm = ChatOllama(model="llama3")
    results = db.similarity_search(query, k=3)
    context = ""
    for i, docs in enumerate(results):
        print(f"---Source {i + 1}--- ")
        source = Path(docs.metadata["source"]).name
        print(source)
        print(docs.page_content[:300])
        context += docs.page_content
        context += "\n\n"

    prompt = f"""
    Answer only what we ask 

    context: {context}

    query : {query}

    """
    response = llm.invoke(prompt)
    print(response.content)
