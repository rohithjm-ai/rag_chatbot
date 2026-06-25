from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

pdf_path = r"C:\rag_chatbot\data\iso27001.pdf"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
loader = PyPDFLoader(pdf_path)
documents = loader.load()
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
db.save_local("faiss_index")
query = "What is Compatibility with other management system standards?"
results = db.similarity_search(query, k=3)
llm = ChatOllama(model="llama3")

context = ""
for docs in results:
    context += docs.page_content
    context += "\n\n"

prompt = f"""
Answer only what we ask 

context: {context}

query : {query}

"""
response = llm.invoke(prompt)
print(response.content)
