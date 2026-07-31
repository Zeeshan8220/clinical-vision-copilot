"""
Builds the ChromaDB vector index from our curated medical knowledge base.
"""

import chromadb
from data.medical_knowledge import DOCUMENTS

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="medical_knowledge")

collection.add(
    ids=[doc["id"] for doc in DOCUMENTS],
    documents=[doc["text"] for doc in DOCUMENTS],
    metadatas=[{"source": doc["source"]} for doc in DOCUMENTS],
)

print(f"Indexed {len(DOCUMENTS)} documents into ChromaDB.")
print(f"Collection count: {collection.count()}")
