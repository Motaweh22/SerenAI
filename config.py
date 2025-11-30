INDEX_DIR = "./faiss_index_pair_v1"
EMBEDDING_MODEL = "all-mpnet-base-v2"
DEVICE = "cuda"       # غيّر لـ "cuda" لو عايز GPU
TOP_K = 5
MAX_DISPLAY_CHARS = 1200

SELFHARM_KEYWORDS = [
    "suicide", "kill myself", "end my life", "hurt myself", "i want to die"
]

ABUSE_KEYWORDS = [
    "he hit me", "he abused", "sexual abuse", "domestic violence"
]
