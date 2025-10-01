
import streamlit as st
import os, json, random

st.set_page_config(page_title="Dr. Anu's Analysis", layout="wide")

BASE_DIR = os.path.dirname(__file__)
DATA_FULL = os.path.join(BASE_DIR, "data", "remedies_full.json")
DATA_MASTER = os.path.join(BASE_DIR, "data", "remedies_master.json")
DATA_FILE = DATA_FULL if os.path.exists(DATA_FULL) else DATA_MASTER

@st.cache_data
def load_remedies():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

remedies = load_remedies()

st.title("Dr. Anu's Analysis")
st.caption("Ephemeral processing — complaint text is not stored. Use clinically only with professional judgment.")

with st.expander("Enter complaint (input hidden after processing)"):
    complaint = st.text_area("Paste patient's complaint here (processed ephemerally)", height=220, key="complaint_input")
    analyze = st.button("Process complaint")

def simple_token_score(q, text):
    qt = set([t for t in q.lower().split() if len(t)>2])
    return sum(1 for t in qt if t in text.lower()) / max(1, len(text.split()))

def compute_scores(complaint_text, remedies):
    scores=[]
    for r in remedies:
        sem = simple_token_score(complaint_text, r.get("full_text",""))
        kb = 0.0
        for rub in r.get("rubrics", []):
            if rub.lower() in complaint_text.lower():
                kb += 2.0
        total = 0.6*sem + 0.4*(kb/(1+kb))
        scores.append({"remedy": r, "score": total, "sem": sem, "kb": kb})
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores

if 'last_results' not in st.session_state:
    st.session_state['last_results'] = None

if 'analyze' in locals() and analyze:
    if not complaint or complaint.strip()=="":
        st.warning("Please enter a complaint.")
    else:
        results = compute_scores(complaint, remedies)
        max_score = max((r['score'] for r in results), default=0.0)
        norm = max(0.0001, max_score)
        for r in results:
            r['percent'] = round((r['score']/norm)*100,1)
            r['percent'] = max(1.0, min(99.9, r['percent']))
        st.session_state['last_results'] = results
        try:
            if 'complaint_input' in st.session_state:
                st.session_state['complaint_input'] = ""
        except Exception:
            pass

if st.session_state.get('last_results'):
    results = st.session_state['last_results']
    options = [f"{item['remedy'].get('name','')} — {item['percent']}%" for item in results[:12]]
    selected = st.selectbox('Suggested remedies (select one to view details)', options)
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
    with st.expander('Why this remedy? (internal breakdown)'):
        st.write('Semantic score (token overlap):', round(chosen_meta.get('sem',0),4))
        st.write('Rule/rubric boost:', round(chosen_meta.get('kb',0),4))
        st.write('Combined percent:', chosen_meta.get('percent'))
        st.write('Matched tags:', chosen_meta.get('tags',[]))
else:
    st.info("No analysis yet. Enter a complaint and click 'Process complaint' to get suggestions.")

st.markdown('---')
st.caption('This tool suggests candidate remedies for practitioner review only. Final prescription is the responsibility of the clinician.')
