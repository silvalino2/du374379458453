import numpy as np
import pickle
import os
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
store = []

STORE_FILE = "store.pkl"

def save_store():
    with open(STORE_FILE, "wb") as f:
        pickle.dump(store, f)

def load_store():
    global store
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "rb") as f:
            store = pickle.load(f)

load_store()  # runs on import — loads saved facts if store.pkl exists

def embed(text: str):
    return list(model.embed([text]))[0]

def add_fact(id: str, text: str, source: str = "", country: str = "Nigeria"):
    embedding = embed(text)
    store.append({"id": id, "text": text, "source": source, "country": country, "embedding": embedding})

def query_facts(question: str, n: int = 3):
    q_embedding = embed(question)
    scored = []
    for item in store:
        sim = np.dot(q_embedding, item["embedding"]) / (
            np.linalg.norm(q_embedding) * np.linalg.norm(item["embedding"])
        )
        scored.append((sim, item["text"], item["source"]))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:n]
