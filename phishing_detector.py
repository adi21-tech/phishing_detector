"""
Phishing URL Detector - step by step
------------------------------------
Setup:
    pip install pandas scikit-learn joblib

Dataset:
    Download a CSV with two columns: url, label
    e.g. Kaggle "Phishing Site URLs" (columns: URL, Label with values good/bad)
    Save it as dataset.csv in the same folder.

Run:
    python phishing_detector.py train
    python phishing_detector.py predict "http://paypal-login.verify-account.xyz/signin"
"""

import re
import sys
import math
from collections import Counter
from urllib.parse import urlparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DATASET = "dataset.csv/phishing_site_urls.csv"
MODEL_FILE = "phishing_model.joblib"

# ---------------------------------------------------------------
# STEP 1: Feature extraction (turn a URL into numbers)
# ---------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "bank", "signin",
    "confirm", "password", "paypal", "wallet", "free", "bonus", "support",
]
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"]


def entropy(text):
    """Randomness of characters; random-looking domains score higher."""
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_features(url):
    url = str(url).strip()
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
    except ValueError:
        # Malformed URL (e.g. stray brackets). Fall back to a safe empty parse.
        parsed = urlparse("http://invalid")
    host = parsed.netloc.split(":")[0].lower()
    path = parsed.path
    query = parsed.query

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": host.count("-"),
        "at_symbol": int("@" in url),
        "digit_count": sum(c.isdigit() for c in url),
        "special_char_count": len(re.findall(r"[^a-zA-Z0-9]", url)),
        "subdomain_count": max(host.count(".") - 1, 0),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "is_https": int(parsed.scheme == "https"),
        "double_slash_in_path": int("//" in path),
        "suspicious_words": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
        "is_shortener": int(any(s in host for s in SHORTENERS)),
        "host_entropy": entropy(host),
        "url_entropy": entropy(url),
    }


# ---------------------------------------------------------------
# STEP 2: Load and clean the dataset
# ---------------------------------------------------------------
LABEL_MAP = {
    "bad": 1, "phishing": 1, "malicious": 1, "1": 1, "-1": 1,
    "good": 0, "benign": 0, "legitimate": 0, "0": 0,
}


def load_data(path=DATASET):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"urls": "url"})
    df = df[["url", "label"]].dropna().drop_duplicates()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df)} rows | phishing: {df['label'].sum()} | legit: {(df['label'] == 0).sum()}")
    return df


# ---------------------------------------------------------------
# STEP 3: Build feature table
# ---------------------------------------------------------------
def build_features(df):
    X = pd.DataFrame([extract_features(u) for u in df["url"]])
    y = df["label"]
    return X, y


# ---------------------------------------------------------------
# STEP 4: Train, evaluate, save
# ---------------------------------------------------------------
def train():
    df = load_data()
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nReport:\n", classification_report(y_test, pred, target_names=["legit", "phishing"]))

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Top features:\n", importance.head(8))

    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_FILE)
    print(f"\nModel saved to {MODEL_FILE}")


# ---------------------------------------------------------------
# STEP 5: Predict on a new URL
# ---------------------------------------------------------------
def predict(url):
    bundle = joblib.load(MODEL_FILE)
    model, columns = bundle["model"], bundle["columns"]
    X = pd.DataFrame([extract_features(url)])[columns]
    prob = model.predict_proba(X)[0][1]
    verdict = "PHISHING" if prob >= 0.5 else "LEGITIMATE"
    print(f"{url}\n-> {verdict} (phishing probability: {prob:.2%})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        print(__doc__)