"""NLP backend: preprocessing, training, evaluation, prediction + explanation."""
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "model.joblib"

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.I)
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
MONEY_RE = re.compile(r"[$€£₹]\s?\d[\d,\.]*")
NUM_RE = re.compile(r"\d+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
# keep words that are strong spam signals even though some lists treat them as stopwords
STOP = set(ENGLISH_STOP_WORDS) - {"free", "now", "only", "act", "win"}


def preprocess(text: str) -> str:
    """Normalise raw email text into a clean token string."""
    text = str(text)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = MONEY_RE.sub(" moneytoken ", text)
    text = NUM_RE.sub(" numtoken ", text)
    text = NON_ALPHA_RE.sub(" ", text.lower())
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 1)


def load_dataset(path=None) -> pd.DataFrame:
    """Load a CSV with (label,text) columns, or the Kaggle SMS format (v1,v2)."""
    if path is None:
        csvs = sorted(DATA_DIR.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError("No CSV found in data/. Run `python generate_dataset.py`.")
        # prefer a user-supplied dataset over the bundled starter one
        path = next((p for p in csvs if p.name != "emails.csv"), csvs[0])
    df = pd.read_csv(path, encoding="latin-1")
    cols = {c.lower(): c for c in df.columns}
    if "v1" in cols and "v2" in cols:
        df = df.rename(columns={cols["v1"]: "label", cols["v2"]: "text"})
    else:
        label_col = next((cols[c] for c in ("label", "category", "class", "spam") if c in cols), None)
        text_col = next((cols[c] for c in ("text", "message", "email", "body", "content") if c in cols), None)
        if not label_col or not text_col:
            raise ValueError("CSV needs a label column and a text column (e.g. label,text).")
        df = df.rename(columns={label_col: "label", text_col: "text"})
    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(str).str.lower().map(
        {"spam": 1, "ham": 0, "1": 1, "0": 0, "legitimate": 0, "not spam": 0}
    )
    df = df.dropna(subset=["label"]).drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)
    df.attrs["source"] = Path(path).name
    return df


def _pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def _metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train(path=None, save=True) -> dict:
    """Train Logistic Regression (deployed) and Naive Bayes (baseline); return a bundle."""
    df = load_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
    )
    lr = _pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_train, y_train)
    nb = _pipeline(MultinomialNB(alpha=0.1)).fit(X_train, y_train)

    results = {}
    for name, model in (("Logistic Regression", lr), ("Naive Bayes", nb)):
        results[name] = _metrics(y_test, model.predict(X_test))

    bundle = {
        "model": lr,
        "results": results,
        "confusion": confusion_matrix(y_test, lr.predict(X_test)).tolist(),
        "n_samples": len(df),
        "n_spam": int(df["label"].sum()),
        "source": df.attrs.get("source", "unknown"),
    }
    if save:
        joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_or_train() -> dict:
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:  # version mismatch etc. -> retrain
            pass
    return train()


def predict(bundle: dict, text: str) -> float:
    """Return spam probability (0-1)."""
    return float(bundle["model"].predict_proba([text])[0][1])


def predict_many(bundle: dict, texts) -> np.ndarray:
    return bundle["model"].predict_proba(list(texts))[:, 1]


def word_weights(bundle: dict) -> dict:
    """Map every vocabulary term -> LR coefficient (positive = spammy)."""
    vec, clf = bundle["model"].named_steps["tfidf"], bundle["model"].named_steps["clf"]
    return dict(zip(vec.get_feature_names_out(), clf.coef_[0]))


def top_features(bundle: dict, n=15):
    w = sorted(word_weights(bundle).items(), key=lambda kv: kv[1])
    return w[-n:][::-1], w[:n]


def explain(bundle: dict, text: str, n=8):
    """Top terms in this email pushing toward spam / ham (tfidf * coefficient)."""
    vec, clf = bundle["model"].named_steps["tfidf"], bundle["model"].named_steps["clf"]
    x = vec.transform([text])
    names = vec.get_feature_names_out()
    contrib = x.toarray()[0] * clf.coef_[0]
    idx = np.argsort(contrib)
    spam = [(names[i], contrib[i]) for i in idx[::-1][:n] if contrib[i] > 0]
    ham = [(names[i], contrib[i]) for i in idx[:n] if contrib[i] < 0]
    return spam, ham


if __name__ == "__main__":
    b = train()
    print(f"Trained on {b['n_samples']} emails ({b['source']})")
    for k, v in b["results"].items():
        print(k, {m: round(s, 3) for m, s in v.items()})

  """Streamlit frontend for the NLP email spam detector."""
import html
import re

import pandas as pd
import streamlit as st

import backend as be

st.set_page_config(page_title="Email Spam Detector", page_icon="📧", layout="wide")


@st.cache_resource(show_spinner="Loading model (trains on first run)...")
def get_bundle():
    return be.load_or_train()


@st.cache_data
def get_weights(_bundle, key):
    return be.word_weights(_bundle)


bundle = get_bundle()
weights = get_weights(bundle, bundle["n_samples"])

EXAMPLES = {
    "— pick an example —": "",
    "Prize scam": "CONGRATULATIONS!!! You have been selected to win a $1000 gift card. Click http://bit.ly/claim-now to claim your prize now!",
    "Phishing": "URGENT: Your PayPal account has been suspended. Verify your password immediately at http://secure-login-verify.xyz or lose access.",
    "Work email": "Hi Priya, can we move tomorrow's meeting to Thursday at 11:00? Let me know what works for you.",
    "Study group": "Hey Rahul, are you coming to the study group tonight? We are covering chapter 4 of the textbook.",
}


def highlight(text: str) -> str:
    """Colour each word red/green by its learned spam weight."""
    out = []
    for tok in re.findall(r"\S+|\s+", text):
        if tok.isspace():
            out.append(tok)
            continue
        key = be.preprocess(tok).split()
        w = weights.get(key[0], 0) if key else 0
        safe = html.escape(tok)
        if w > 0.4:
            out.append(f"<span style='background:rgba(255,75,75,.30);border-radius:4px;padding:0 2px'>{safe}</span>")
        elif w < -0.4:
            out.append(f"<span style='background:rgba(40,190,100,.28);border-radius:4px;padding:0 2px'>{safe}</span>")
        else:
            out.append(safe)
    return "".join(out).replace("\n", "<br>")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    threshold = st.slider("Spam threshold", 0.10, 0.95, 0.50, 0.05,
                          help="Emails with spam probability above this are flagged as spam.")
    st.markdown("---")
    st.caption(f"Model: TF-IDF (1–2 grams) + Logistic Regression")
    st.caption(f"Trained on **{bundle['n_samples']}** emails from `{bundle['source']}`")
    if bundle["source"] == "emails.csv":
        st.info("Using the small bundled starter dataset. For real-world accuracy, add a larger dataset to `data/` (see README).")

st.title("📧 Email Spam Detector")
st.caption("An NLP app using text preprocessing, TF-IDF features and a Logistic Regression classifier.")

tab1, tab2, tab3 = st.tabs(["🔍 Check an email", "📂 Batch (CSV)", "📊 Model insights"])

# ---------- Tab 1: single email ----------
with tab1:
    choice = st.selectbox("Try an example", list(EXAMPLES))
    text = st.text_area("Paste email text", value=EXAMPLES[choice], height=200,
                        placeholder="Paste the subject and body of an email here...")
    if st.button("Analyze", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
        else:
            p = be.predict(bundle, text)
            c1, c2 = st.columns([1, 2])
            with c1:
                if p >= threshold:
                    st.error("🚨 SPAM")
                else:
                    st.success("✅ NOT SPAM (ham)")
                st.metric("Spam probability", f"{p:.1%}")
                st.progress(min(max(p, 0.0), 1.0))
            with c2:
                spam_terms, ham_terms = be.explain(bundle, text)
                a, b = st.columns(2)
                a.markdown("**Spam signals**")
                a.write(", ".join(f"`{t}`" for t, _ in spam_terms) or "_none_")
                b.markdown("**Legitimate signals**")
                b.write(", ".join(f"`{t}`" for t, _ in ham_terms) or "_none_")
            st.markdown("##### Highlighted text")
            st.markdown(highlight(text), unsafe_allow_html=True)
            st.caption("🔴 pushes toward spam · 🟢 pushes toward legitimate")

# ---------- Tab 2: batch ----------
with tab2:
    st.write("Upload a CSV with a column containing email text (`text`, `message`, `email`, `body`).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up, encoding="latin-1")
        col = next((c for c in df.columns if c.lower() in ("text", "message", "email", "body", "content", "v2")), None)
        if col is None:
            st.error("Couldn't find a text column. Rename it to `text`.")
        else:
            df["spam_probability"] = be.predict_many(bundle, df[col].fillna("").astype(str))
            df["prediction"] = (df["spam_probability"] >= threshold).map({True: "spam", False: "ham"})
            m1, m2, m3 = st.columns(3)
            m1.metric("Emails", len(df))
            m2.metric("Flagged spam", int((df["prediction"] == "spam").sum()))
            m3.metric("Spam rate", f"{(df['prediction'] == 'spam').mean():.1%}")
            st.dataframe(df, use_container_width=True)
            st.download_button("Download results", df.to_csv(index=False).encode(),
                               "spam_predictions.csv", "text/csv")

# ---------- Tab 3: insights ----------
with tab3:
    st.subheader("Evaluation on held-out 20% test set")
    st.dataframe(pd.DataFrame(bundle["results"]).T.style.format("{:.3f}"), use_container_width=True)
    cm = bundle["confusion"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Confusion matrix** (rows = actual, cols = predicted)")
        st.dataframe(pd.DataFrame(cm, index=["actual ham", "actual spam"], columns=["pred ham", "pred spam"]))
    spam_top, ham_top = be.top_features(bundle, 12)
    with c2:
        st.markdown("**Most spam-indicating terms**")
        st.bar_chart(pd.Series(dict(spam_top)))
    st.markdown("**Most legitimate-indicating terms**")
    st.bar_chart(pd.Series({k: abs(v) for k, v in ham_top}))

    with st.expander("🔁 Retrain on your own data"):
        st.write("Upload a CSV with `label` (spam/ham) and `text` columns, or the Kaggle SMS format (`v1`,`v2`).")
        train_file = st.file_uploader("Training CSV", type="csv", key="train")
        if train_file and st.button("Retrain model"):
            tmp = be.DATA_DIR / "uploaded.csv"
            tmp.write_bytes(train_file.getvalue())
            try:
                be.train(tmp)
                st.cache_resource.clear()
                st.cache_data.clear()
                st.success("Retrained! Reloading...")
                st.rerun()
            except Exception as e:
                tmp.unlink(missing_ok=True)
                st.error(f"Training failed: {e}")
