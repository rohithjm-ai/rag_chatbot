# this code is only for testing llama is installed or not
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")
response = llm.invoke("What is Rag?")
print(response.content)
