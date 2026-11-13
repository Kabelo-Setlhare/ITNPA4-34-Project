# Run with:  streamlit run app.py      (train the model first: python train_model.py)
import os
import joblib
import streamlit as st
from text_utils import clean_text

LABELS = {1: "artificial intelligence", 0: "classical research"}

st.title("Researchly Research Abstract Classifier")
st.write("This app classifies research abstracts as either AI-related or classical research.")


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abstract_classifier.pkl")
    return joblib.load(path)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Run `python train_model.py` in this folder first.")
    st.stop()

abstract = st.text_area("Enter research abstract:", height=250)

if st.button("Classify Abstract"):
    if not abstract.strip():
        st.warning("Please enter an abstract first.")
    else:
        cleaned = clean_text(abstract)
        pred = int(model.predict([cleaned])[0])
        prob = float(model.predict_proba([cleaned])[0][pred])
        st.subheader("Prediction")
        st.success(f"The abstract is classified as: {LABELS[pred]}")
        st.caption(f"Model confidence: {prob:.0%}")
