from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

pdf_path = r"C:\rag_chatbot\data\iso27001.pdf"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
loader = PyPDFLoader(pdf_path)
documents = loader.load()
chunks = splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embeddings.embed_query("who at is cat?")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("PAGES : ", len(documents))
print(documents[0].page_content[:500])
print("Chunk Size : ", len(chunks))
print(chunks[0].page_content)
print("Vector Size : ", len(vector))
print("First 10 values after Embedding : ", vector[:10])
print("Vector Database is Created")
