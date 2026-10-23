"""
Phishing URL Detector v2 - handcrafted features + character n-gram TF-IDF + tuned threshold
-----------------------------------------------------------------------------------------
Setup:
    pip install pandas scikit-learn joblib scipy

Dataset:
    CSV with two columns: url, label (Kaggle "Phishing Site URLs": URL, Label good/bad)

Run:
    python phishing_detector.py train
    python phishing_detector.py predict "http://paypal-login.verify-account.xyz/signin"
"""

import re
import sys
import math
import time
from collections import Counter
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split

DATASET = "dataset.csv/phishing_site_urls.csv"
MODEL_FILE = "phishing_model_v2.joblib"
TARGET_PRECISION = 0.85  # threshold is chosen to maximise recall while keeping precision >= this

# ---------------------------------------------------------------
# STEP 1: Handcrafted features (same as before)
# ---------------------------------------------------------------
SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "bank", "signin",
    "confirm", "password", "paypal", "wallet", "free", "bonus", "support",
]
SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly"]


def entropy(text):
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
    df["url"] = df["url"].astype(str)
    print(f"Loaded {len(df)} rows | phishing: {df['label'].sum()} | legit: {(df['label'] == 0).sum()}")
    return df


def build_numeric(urls):
    return pd.DataFrame([extract_features(u) for u in urls])


# ---------------------------------------------------------------
# STEP 3: Choose the decision threshold on the VALIDATION set
# ---------------------------------------------------------------
def choose_threshold(y_val, prob_val, target_precision=TARGET_PRECISION):
    prec, rec, thr = precision_recall_curve(y_val, prob_val)
    prec, rec = prec[:-1], rec[:-1]  # align with thr
    ok = prec >= target_precision
    if ok.any():
        best = np.argmax(np.where(ok, rec, -1))  # highest recall that still meets precision target
    else:
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        best = np.argmax(f1)
    return float(thr[best]), float(prec[best]), float(rec[best])


# ---------------------------------------------------------------
# STEP 4: Train, evaluate, save
# ---------------------------------------------------------------
def train():
    t0 = time.time()
    df = load_data()
    urls, y = df["url"].values, df["label"].values

    # 60% train / 20% validation (threshold tuning) / 20% test (final honest numbers)
    u_train, u_tmp, y_train, y_tmp = train_test_split(
        urls, y, test_size=0.4, random_state=42, stratify=y
    )
    u_val, u_test, y_val, y_test = train_test_split(
        u_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    # --- Model A: Random Forest on handcrafted features ---
    print("\nExtracting handcrafted features...")
    X_train_n, X_val_n, X_test_n = (build_numeric(u) for u in (u_train, u_val, u_test))
    columns = list(X_train_n.columns)

    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train_n, y_train)

    # --- Model B: Logistic Regression on character n-gram TF-IDF ---
    print("Training TF-IDF + Logistic Regression...")
    tfidf = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), max_features=300000,
        sublinear_tf=True, lowercase=True, dtype=np.float32,
    )
    T_train = tfidf.fit_transform(u_train)
    T_val, T_test = tfidf.transform(u_val), tfidf.transform(u_test)

    lr = LogisticRegression(
        C=5.0, class_weight="balanced", max_iter=1000, solver="liblinear"
    )
    lr.fit(T_train, y_train)

    # --- Compare models, then ensemble (average probabilities) ---
    p_rf_val, p_lr_val = rf.predict_proba(X_val_n)[:, 1], lr.predict_proba(T_val)[:, 1]
    p_rf_test, p_lr_test = rf.predict_proba(X_test_n)[:, 1], lr.predict_proba(T_test)[:, 1]
    p_val = (p_rf_val + p_lr_val) / 2
    p_test = (p_rf_test + p_lr_test) / 2

    print("\n=== Test accuracy at default 0.5 threshold ===")
    print(f"Random Forest (handcrafted): {accuracy_score(y_test, p_rf_test >= 0.5):.4f}")
    print(f"TF-IDF + LogReg (n-grams)  : {accuracy_score(y_test, p_lr_test >= 0.5):.4f}")
    print(f"Ensemble (average)         : {accuracy_score(y_test, p_test >= 0.5):.4f}")

    # --- Threshold tuning on validation, evaluation on untouched test set ---
    threshold, v_prec, v_rec = choose_threshold(y_val, p_val)
    print(f"\nChosen threshold (from validation): {threshold:.3f} "
          f"(val precision {v_prec:.2f}, val recall {v_rec:.2f})")

    pred = (p_test >= threshold).astype(int)
    print(f"\n=== FINAL TEST RESULTS (threshold {threshold:.3f}) ===")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))
    print("\nReport:\n", classification_report(y_test, pred, target_names=["legit", "phishing"]))

    importance = pd.Series(rf.feature_importances_, index=columns).sort_values(ascending=False)
    print("Top handcrafted features:\n", importance.head(8))

    joblib.dump(
        {"rf": rf, "tfidf": tfidf, "lr": lr, "columns": columns, "threshold": threshold},
        MODEL_FILE,
    )
    print(f"\nModel saved to {MODEL_FILE} | total time: {time.time() - t0:.0f}s")


# ---------------------------------------------------------------
# STEP 5: Predict on a new URL
# ---------------------------------------------------------------
def predict(url):
    b = joblib.load(MODEL_FILE)
    X = pd.DataFrame([extract_features(url)])[b["columns"]]
    p_rf = b["rf"].predict_proba(X)[0][1]
    p_lr = b["lr"].predict_proba(b["tfidf"].transform([url]))[0][1]
    prob = (p_rf + p_lr) / 2
    verdict = "PHISHING" if prob >= b["threshold"] else "LEGITIMATE"
    print(f"{url}\n-> {verdict} (phishing probability: {prob:.2%}, threshold: {b['threshold']:.2f})")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) >= 3 and sys.argv[1] == "predict":
        predict(sys.argv[2])
    else:
        print(__doc__)