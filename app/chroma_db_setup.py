import chromadb
from pathlib import Path


EMBEDDINGS_PATH=Path( Path(__file__).resolve().parent / "chroma_db" )

client = chromadb.PersistentClient(str(EMBEDDINGS_PATH))

collection = client.get_or_create_collection(name = "face_embeddings_all", embedding_function=None)
print(collection.get())