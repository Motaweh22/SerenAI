import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from config import INDEX_DIR, EMBEDDING_MODEL, DEVICE

class PairIndexLoader:
    def __init__(self):
        self.index = None
        self.texts = None
        self.meta = None
        self.embedder = None

    def load(self):
        index_path = os.path.join(INDEX_DIR, "index.faiss")
        metadata_path = os.path.join(INDEX_DIR, "metadatas.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index not found at " + index_path)

        self.index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            meta = pickle.load(f)

        self.texts = meta["texts"]
        self.meta = meta["metadatas"]

        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

        print(f"[Loader] Loaded index with {self.index.ntotal} vectors")
        return self

loader = PairIndexLoader().load()
