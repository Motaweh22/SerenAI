import faiss
from loader import loader
from config import TOP_K

class PairRetriever:
    def __init__(self, index_loader=loader):
        self.index = index_loader.index
        self.texts = index_loader.texts
        self.meta = index_loader.meta
        self.embedder = index_loader.embedder

    def retrieve(self, query, k=TOP_K):
        q_emb = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            results.append({
                "score": float(score),
                "pair_text": self.texts[idx],
                "metadata": self.meta[idx]
            })
        return results

retriever = PairRetriever()
