"""NLP backend: preprocessing, training, evaluation, prediction + explanation."""
import random
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
STARTER_NAME = "built-in starter data"

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


def starter_dataset() -> pd.DataFrame:
    """Small synthetic dataset, used when no CSV exists in data/."""
    rnd = random.Random(42)
    pick = rnd.choice
    prizes = ["a $1000 gift card", "an iPhone 15", "$5,000 cash", "a free cruise", "a brand new laptop"]
    urls = ["http://bit.ly/claim-now", "www.win-big-prizes.biz/claim", "http://secure-login-verify.xyz", "http://free-offer.top/go"]
    drugs = ["cheap meds", "discount pills", "V1agra and C1alis", "prescription drugs without prescription"]
    banks = ["PayPal", "your bank", "Amazon", "Netflix", "Apple ID", "HDFC Bank"]
    names = ["Priya", "Rahul", "Anita", "Sam", "Neha", "Karan", "Maria", "John", "Aisha", "Dev"]
    topics = ["the project report", "tomorrow's meeting", "the quarterly budget", "the lab assignment",
              "the client presentation", "the design review"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    spam = [
        lambda: f"CONGRATULATIONS!!! You have been selected to win {pick(prizes)}. Click {pick(urls)} to claim your prize now!",
        lambda: f"URGENT: Your {pick(banks)} account has been suspended. Verify your password immediately at {pick(urls)} or lose access.",
        lambda: f"Get {pick(drugs)} online. No prescription needed. Order now and save 80%! Limited time offer.",
        lambda: f"Make ${rnd.randint(500, 5000)} per week working from home! No experience needed. Sign up free: {pick(urls)}",
        lambda: f"Dear friend, I am a prince with {rnd.randint(2, 50)} million dollars to transfer. Reply with your bank details to receive your share.",
        lambda: f"FREE entry in our weekly draw! Text WIN to {rnd.randint(80000, 89999)} now. T&Cs apply. Act fast, offer ends tonight!",
        lambda: f"Final notice: you owe unpaid taxes. Pay now via gift cards to avoid arrest. Call {rnd.randint(1000000000, 9999999999)}.",
        lambda: f"Lowest mortgage rates guaranteed! Refinance today, bad credit OK. Apply here {pick(urls)} - no obligation, 100% free.",
        lambda: f"You've got a package waiting. Confirm delivery fee of ${rnd.randint(2, 9)}.99 at {pick(urls)} within 24 hours.",
        lambda: f"Hot singles in your area want to meet you! Click here {pick(urls)}. Unsubscribe not available. Buy now, winner!",
        lambda: f"Earn bitcoin fast! Double your investment in {rnd.randint(2, 10)} days guaranteed. Join our exclusive crypto club {pick(urls)}",
        lambda: f"Claim your refund of ${rnd.randint(100, 999)}. Your {pick(banks)} refund is pending, enter card number and CVV to receive it.",
    ]
    ham = [
        lambda: f"Hi {pick(names)}, can we move {pick(topics)} to {pick(days)} at {rnd.randint(9, 17)}:00? Let me know what works for you.",
        lambda: f"Thanks for sending {pick(topics)}. I reviewed it and left a few comments in the shared document. Talk soon, {pick(names)}.",
        lambda: f"Reminder: our team lunch is on {pick(days)}. Please tell me about any dietary preferences before then.",
        lambda: f"Hey {pick(names)}, are you coming to the study group tonight? We are covering chapter {rnd.randint(2, 9)} of the textbook.",
        lambda: f"Attached is the invoice for last month's work. Please let me know if you have any questions about {pick(topics)}.",
        lambda: f"Good morning team, the standup is at {rnd.randint(9, 11)}:30 today. Please update your tickets before the call.",
        lambda: f"Hi {pick(names)}, I'll be late today because of traffic. Please start without me and I will join in twenty minutes.",
        lambda: f"Your order has shipped and should arrive by {pick(days)}. You can track it from your account page whenever you like.",
        lambda: f"Dear professor, I would like to ask for an extension on {pick(topics)} because I was unwell. Thank you for understanding.",
        lambda: f"Happy birthday {pick(names)}! Hope you have a wonderful day. Dinner at my place on {pick(days)}?",
        lambda: f"Minutes from {pick(topics)}: we agreed on next steps and {pick(names)} will follow up with the vendor this week.",
        lambda: f"Could you review my pull request when you get a chance? It fixes the login bug we discussed on {pick(days)}.",
    ]

    def mutate(t):
        r = rnd.random()
        return t.upper() if r < 0.15 else t.lower() if r < 0.30 else t

    rows = [(mutate(pick(spam)()), 1) for _ in range(500)] + [(mutate(pick(ham)()), 0) for _ in range(500)]
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates(subset="text")
    return df.sample(frac=1, random_state=1).reset_index(drop=True)


def load_dataset(path=None) -> pd.DataFrame:
    """Load a CSV with (label,text) columns, or the Kaggle SMS format (v1,v2).
    With no path: use the newest-named CSV in data/ (user data preferred over emails.csv),
    or fall back to the built-in starter dataset."""
    if path is None:
        csvs = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
        if not csvs:
            df = starter_dataset()
            df.attrs["source"] = STARTER_NAME
            return df
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
        try:
            joblib.dump(bundle, MODEL_PATH)
        except Exception:
            pass  # read-only filesystem: just keep the model in memory
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
