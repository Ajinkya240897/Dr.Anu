"""
scripts/build_index.py
Build embeddings and FAISS index from data/remedies_full.json (if present) or data/remedies_master.json.
Saves embeddings.npy and faiss.index to data/.
Uses the smaller 'all-MiniLM-L6-v2' model for reliability on hosted runners.
"""
import os
import json
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FULL = os.path.join(BASE_DIR, "data", "remedies_full.json")
DATA_MASTER = os.path.join(BASE_DIR, "data", "remedies_master.json")
DATA_FILE = DATA_FULL if os.path.exists(DATA_FULL) else DATA_MASTER

OUT_EMB = os.path.join(BASE_DIR, "data", "embeddings.npy")
OUT_IDX = os.path.join(BASE_DIR, "data", "faiss.index")
OUT_MAP = os.path.join(BASE_DIR, "data", "remedies_full_index_map.json")

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d.get("full_text", "")[:4000] for d in docs]

    if len(texts) == 0:
        print("No documents to encode. Exiting.")
        return

    # Use the smaller, more robust MiniLM model for hosted runners
    model_name = "all-MiniLM-L6-v2"
    print("Loading model", model_name)
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts)} documents...")
    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embs = np.array(embs).astype("float32")

    print("Saving embeddings to", OUT_EMB)
    np.save(OUT_EMB, embs)

    dim = embs.shape[1]
    print("Embedding dim:", dim)

    print("Normalizing vectors and building FAISS index...")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, OUT_IDX)
    print("Saved FAISS index to", OUT_IDX)

    print("Saving index map to", OUT_MAP)
    with open(OUT_MAP, "w", encoding="utf-8") as f:
        json.dump([d.get("name", "") for d in docs], f, indent=2, ensure_ascii=False)

    print("Build complete.")

if __name__ == "__main__":
    main()
