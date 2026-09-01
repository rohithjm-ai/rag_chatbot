from pathlib import Path

folder_path = Path(r"C:\rag_chatbot\data")
pdf_files = folder_path.glob("*pdf")
for pdf in pdf_files:
    print(pdf)
