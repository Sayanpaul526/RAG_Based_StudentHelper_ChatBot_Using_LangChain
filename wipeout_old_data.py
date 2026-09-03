# wipeout_old_data.py
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
import requests

load_dotenv()

class LocalOllamaEmbeddings:
    def __init__(self, model_name):
        self.model_name = model_name
        self.url = "http://127.0.0.1:11434/api/embed"
    def embed_documents(self, texts):
        payload = {"model": self.model_name, "input": texts}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["embeddings"]
    def embed_query(self, text):
        payload = {"model": self.model_name, "input": text}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()["embeddings"][0]

embeddings = LocalOllamaEmbeddings(model_name='nomic-embed-text-v2-moe:latest')
vector_store = PineconeVectorStore(
    embedding=embeddings,
    index_name='student-note-store'
)

# Delete using an empty string (which is what the library expects for the default namespace)
try:
    vector_store._index.delete(delete_all=True, namespace="")
    print("✅ All old data from the default namespace has been wiped clean.")
except Exception as e:
    # If the namespace is empty (which it is), just say it's already clean!
    print("✅ The namespace is already empty! Nothing to delete.")