"""Shared text denoising used by the notebooks and the deployed app."""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+")
_HTML = re.compile(r"<[^>]+>")
_NON_LETTER = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text):
    """Lowercase, strip URLs/emails/HTML/numbers/punctuation, drop stop words and 1-2 letter tokens."""
    text = str(text).lower()
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _HTML.sub(" ", text)
    text = _NON_LETTER.sub(" ", text)
    tokens = [t for t in _SPACES.sub(" ", text).split()
              if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    return " ".join(tokens)
