# streamlit_app.py (FAISS-enabled with lazy model load using MiniLM)
import streamlit as st
import os
import json
import random
import numpy as np

st.set_page_config(page_title="Dr. Anu's Analysis", layout="wide")
BASE_DIR = os.path.dirname(__file__)
DATA_FULL = os.path.join(BASE_DIR, "data", "remedies_full.json")
DATA_MASTER = os.path.join(BASE_DIR, "data", "remedies_master.json")
EMB_FILE = os.path.join(BASE_DIR, "data", "embeddings.npy")
FAISS_FILE = os.path.join(BASE_DIR, "data", "faiss.index")
INDEX_MAP = os.path.join(BASE_DIR, "data", "remedies_full_index_map.json")

# Try to import heavy libs, but tolerate absence
has_faiss = False
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    has_faiss = True
except Exception:
    has_faiss = False

@st.cache_data
def load_remedies():
    path = DATA_FULL if os.path.exists(DATA_FULL) else DATA_MASTER
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

remedies = load_remedies()

# Top-level debug info
st.title("Dr. Anu's Analysis")
st.caption("Ephemeral processing — complaint text is not stored. Use clinically only with professional judgment.")
st.markdown("**Debug info:**")
cols = st.columns([1,1,1,1])
cols[0].metric("Remedies loaded", len(remedies))
cols[1].metric("Embeddings file", "Yes" if os.path.exists(EMB_FILE) else "No")
cols[2].metric("FAISS index", "Yes" if os.path.exists(FAISS_FILE) else "No")
cols[3].metric("Faiss lib available", str(has_faiss))

# Attempt to load index & embeddings (but don't load model yet)
index = None
embeddings = None
index_map = None
use_faiss = False

if os.path.exists(EMB_FILE) and os.path.exists(FAISS_FILE) and has_faiss:
    try:
        embeddings = np.load(EMB_FILE)
        index = faiss.read_index(FAISS_FILE)
        if os.path.exists(INDEX_MAP):
            with open(INDEX_MAP, "r", encoding="utf-8") as f:
                index_map = json.load(f)
        use_faiss = True
        st.info("FAISS index and embeddings found. Model will be loaded lazily when needed (MiniLM).")
    except Exception as e:
        st.warning(f"Failed to load FAISS/index: {e}. Falling back to token scoring.")
        use_faiss = False
else:
    st.info("FAISS or embeddings not available — using token-overlap fallback scoring.")

# Input UI (ephemeral)
with st.expander("Enter complaint (input hidden after processing)"):
    complaint_text = st.text_area("Paste patient's complaint here (processed ephemerally)", height=240, key="complaint_input")
    analyze = st.button("Process complaint")

# Lazy model holder
_model = None
def get_model():
    global _model
    if _model is None:
        try:
            # MiniLM is smaller and avoids meta-tensor issues on hosted platforms
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            _model = None
    return _model

# Scoring helpers
def token_score(q, text):
    qt = set([t for t in q.lower().split() if len(t)>2])
    if not text:
        return 0.0
    text_l = text.lower()
    s = sum(1 for t in qt if t in text_l)
    return s / max(1, len(text_l.split()))

def semantic_search_scores(query, top_k=20):
    model = get_model()
    if model is None or index is None or embeddings is None:
        return []
    # encode query
    q_emb = model.encode(query, convert_to_numpy=True).astype("float32")
    # normalize (FAISS index uses normalized vectors)
    faiss.normalize_L2(np.expand_dims(q_emb, 0))
    faiss.normalize_L2(embeddings)
    D, I = index.search(np.expand_dims(q_emb, 0), top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"idx": int(idx), "score": float(score)})
    return results

def compute_candidates(complaint):
    if not complaint or complaint.strip() == "":
        return []
    if use_faiss and _model is not None or (use_faiss and get_model() is not None):
        model_ok = get_model() is not None
        if model_ok:
            hits = semantic_search_scores(complaint, top_k=min(50, len(remedies)))
            if len(hits) == 0:
                return []
            max_score = max(h['score'] for h in hits) or 1.0
            results = []
            seen = set()
            for h in hits:
                idx = h['idx']
                if idx in seen or idx >= len(remedies):
                    continue
                seen.add(idx)
                rem = remedies[idx] if idx < len(remedies) else None
                pct = (h['score'] / max_score) * 100.0
                kb = 0.0
                for rub in (rem.get("rubrics",[]) if rem else []):
                    if rub.lower() in complaint.lower():
                        kb += 8.0
                pct = min(99.9, pct + kb)
                results.append({"remedy": rem, "percent": round(pct,1), "score": h['score'], "kb": kb})
            return results
        else:
            # model failed to load — fallback
            pass

    # fallback token-overlap scorer
    scored = []
    for r in remedies:
        s = token_score(complaint, r.get("full_text",""))
        kb = 0.0
        for rub in r.get("rubrics", []):
            if rub.lower() in complaint.lower():
                kb += 0.02
        total = 0.7 * s + 0.3 * kb
        scored.append({"remedy": r, "percent": 0.0, "score": total, "kb": kb})
    if len(scored) == 0:
        return []
    maxs = max(item["score"] for item in scored) or 1.0
    for item in scored:
        item["percent"] = round((item["score"]/maxs)*100,1)
    scored = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored[:50]

# Results display and ephemeral handling
if 'last_results' not in st.session_state:
    st.session_state['last_results'] = None

if 'analyze' in locals() and analyze:
    if not complaint_text or complaint_text.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        results = compute_candidates(complaint_text)
        if not results:
            st.warning("No candidates found. Check ingestion/index or try simpler wording.")
        st.session_state['last_results'] = results
        try:
            if 'complaint_input' in st.session_state:
                st.session_state['complaint_input'] = ""
        except Exception:
            pass

if st.session_state.get('last_results'):
    results = st.session_state['last_results']
    options = [f"{r['remedy'].get('name','Unknown')} — {r.get('percent',0)}%" for r in results[:20]]
    selected = st.selectbox("Suggested remedies (select one to view details)", options)
    sel_idx = options.index(selected)
    chosen = results[sel_idx]['remedy']
    chosen_meta = results[sel_idx]
    st.markdown(f"### Selected remedy: {chosen.get('name','')} — {chosen_meta.get('percent')}%")
    def show_block(title, content):
        st.markdown(f"**{title}**")
        if isinstance(content, list):
            for c in content:
                st.write('-', c)
        else:
            st.write(content if content else '—')
    show_block('Key Characteristics', chosen.get('key_characteristics_desc', chosen.get('key_characteristics', [])))
    show_block('Physical Symptoms', chosen.get('physical_symptoms_desc', chosen.get('physical_symptoms', [])))
    show_block('Mental Symptoms', chosen.get('mental_symptoms_desc', chosen.get('mental_symptoms', [])))
    show_block('Thermal', chosen.get('thermal_desc', chosen.get('thermal','')))
    modalities = chosen.get('modalities', {})
    mod_lines = []
    for k, v in modalities.items():
        if isinstance(v, list):
            mod_lines.append(f"{k.capitalize()}: {', '.join(v)}")
        else:
            mod_lines.append(f"{k.capitalize()}: {v}")
    show_block('Modalities', mod_lines)
    with st.expander("Why this remedy? (internal breakdown)"):
        st.write("Raw score:", chosen_meta.get("score"))
        st.write("Rubric boost:", chosen_meta.get("kb"))
        st.write("Percent shown:", chosen_meta.get("percent"))
else:
    st.info("No analysis yet. Enter a complaint and click 'Process complaint' to get suggestions.")

st.markdown("---")
st.caption("This tool suggests candidate remedies for practitioner review only. Final prescription is the responsibility of the clinician.")
