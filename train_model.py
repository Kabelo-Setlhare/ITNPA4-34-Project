"""Trains the final abstract classifier (the configuration chosen in notebooks/Q2) and saves
abstract_classifier.pkl next to this file. Run:  python train_model.py"""
import json, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from text_utils import clean_text

here = os.path.dirname(os.path.abspath(__file__))
df = pd.DataFrame(json.load(open(os.path.join(here, "..", "data", "research_abstracts.json")))).reset_index(drop=True)
df = df.drop_duplicates(subset="Abstract").reset_index(drop=True)
df["clean"] = df["Abstract"].apply(clean_text)
df["target"] = (df["Label"] == "ai").astype(int)          # 1 = artificial intelligence

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["target"], test_size=0.2, stratify=df["target"], random_state=42)

# Tuned Logistic Regression (best by cross-validated F1 in the Q2 notebook)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=2, sublinear_tf=True, ngram_range=(1, 1))),
    ("clf", LogisticRegression(C=30, max_iter=2000, class_weight="balanced")),
])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print("Test accuracy:", round(accuracy_score(y_test, pred), 3), "| AI-class F1:", round(f1_score(y_test, pred), 3))
joblib.dump(pipe, os.path.join(here, "abstract_classifier.pkl"))
print("Saved abstract_classifier.pkl")
