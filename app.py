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
