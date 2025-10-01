"""
scripts/build_index.py
Build embeddings and FAISS index from remedies_full.json (if present) or remedies_master.json.
Saves embeddings.npy and faiss.index to data/.
"""
import os, json, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_FULL = os.path.join(BASE_DIR, "data", "remedies_full.json")
DATA_MASTER = os.path.join(BASE_DIR, "data", "remedies_master.json")
DATA_FILE = DATA_FULL if os.path.exists(DATA_FULL) else DATA_MASTER

OUT_EMB = os.path.join(BASE_DIR, "data", "embeddings.npy")
OUT_IDX = os.path.join(BASE_DIR, "data", "faiss.index")

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        docs = json.load(f)
    texts = [d.get("full_text","")[:4000] for d in docs]

    # Pick embedding model (large = all-mpnet-base-v2, small/faster = all-MiniLM-L6-v2)
    model_name = "all-mpnet-base-v2"
    print("Loading model", model_name)
    model = SentenceTransformer(model_name)

    embs = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embs = np.array(embs).astype("float32")

    np.save(OUT_EMB, embs)

    dim = embs.shape[1]
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, OUT_IDX)

    print("Saved embeddings and FAISS index to data/")
    with open(os.path.join(BASE_DIR, "data", "remedies_full_index_map.json"), "w", encoding="utf-8") as f:
        json.dump([d.get("name","") for d in docs], f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()
