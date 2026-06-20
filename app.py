from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_path = r"C:\rag_chatbot\data\iso27001.pdf"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
loader = PyPDFLoader(pdf_path)
documents = loader.load()
chunks = splitter.split_documents(documents)
print("PAGES : ", len(documents))
print(documents[0].page_content[:500])
print("Chunk Size : ", len(chunks))
print(chunks[0].page_content)
