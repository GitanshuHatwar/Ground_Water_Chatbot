import faiss
import os
import pickle
from langchain.vectorstores import FAISS

def save_faiss(store, path="faiss_index"):
    store.save_local(path)

def load_faiss(path="faiss_index"):
    if not os.path.exists(path):
        return None
    return FAISS.load_local(path, embeddings=None, allow_dangerous_deserialization=True)
