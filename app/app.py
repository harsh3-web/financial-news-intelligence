import sys, time
from pathlib import Path
import joblib, torch, pandas as pd, streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.data_utils import normalize_numbers          # same cleaning as training

MODELS = ROOT / "models"
LABELS = ["negative", "neutral", "positive"]
ICON = {"negative": "🔴", "neutral": "⚪", "positive": "🟢"}
EXAMPLES = [
    "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn in the corresponding period in 2007 .",
    "The company reported a net loss of EUR 2.3 million and cut its full-year outlook .",
    "The board will meet on Thursday to discuss the annual report .",
    "Operating profit did not rise .",
]

st.set_page_config(page_title="Financial News Sentiment", page_icon="📈", layout="centered")

@st.cache_resource
def load_models():
    tok = BertTokenizer.from_pretrained(MODELS / "finbert")
    fb = BertForSequenceClassification.from_pretrained(MODELS / "finbert").eval()
    vec = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    lr = joblib.load(MODELS / "tfidf_lr.joblib")
    return tok, fb, vec, lr

tok, fb, vec, lr = load_models()

def finbert_probs(text):
    enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        p = fb(**enc).logits.softmax(dim=-1)[0].numpy()
    return dict(zip(LABELS, p))

def tfidf_probs(text):
    p = lr.predict_proba(vec.transform([text]))[0]
    return dict(zip(lr.classes_, p))

st.title("📈 Financial News Sentiment")
st.caption("Fine-tuned FinBERT (test Macro-F1 0.925) compared with a TF-IDF + Logistic Regression baseline (0.806). "
           "Trained on Financial PhraseBank.")

example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
text = st.text_area("Financial news sentence", value="" if example.startswith("(") else example, height=100)

if st.button("Analyze", type="primary") and text.strip():
    clean = normalize_numbers(text.strip())

    t0 = time.perf_counter(); fb_p = finbert_probs(clean); fb_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); tf_p = tfidf_probs(clean);   tf_ms = (time.perf_counter() - t0) * 1000

    fb_label = max(fb_p, key=fb_p.get)
    tf_label = max(tf_p, key=tf_p.get)

    c1, c2 = st.columns(2)
    c1.metric("FinBERT", f"{ICON[fb_label]} {fb_label}", f"{fb_p[fb_label]:.0%} confident · {fb_ms:.0f} ms", delta_color="off")
    c2.metric("TF-IDF baseline", f"{ICON[tf_label]} {tf_label}", f"{tf_p[tf_label]:.0%} confident · {tf_ms:.1f} ms", delta_color="off")

    if fb_p[fb_label] < 0.6:
        st.warning("FinBERT is unsure about this sentence. Treat the label with caution.")
    if fb_label != tf_label:
        st.info("The two models disagree. FinBERT is usually right, but check the limitations below.")

    st.subheader("Probabilities per class")
    st.bar_chart(pd.DataFrame({"FinBERT": fb_p, "TF-IDF": tf_p}).loc[LABELS], stack=False)

with st.expander("Known limitations (from the error analysis)"):
    st.markdown(
        "- Often **ignores negation**: *'profit did not rise'* can come out positive.\n"
        "- **'Bad word, good news'** confuses it: *'loss narrowed'*, *'net debt decreased'*.\n"
        "- Trained on English news about mostly Nordic companies; other styles (tweets, headlines) are less reliable.\n"
        "- Sentiment is **not a market prediction**: news mood did not predict next-day SPY moves (Phase 12). Not investment advice."
    )