""" Sentence embedding """
# TODO: Add dense search with sentence-transformers
# TODO: Add preprocessing to handle the twitter username
from typing import List

import torch
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-2021-124m"


def cosine_similarity(v_a, v_b, eps: float = 1e-5):
    return dot(v_a, v_b) / (norm(v_b) * norm(v_b) + eps)


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class SentenceEmbedding:

    def __init__(self, model: str = None, **kwargs):
        self.model = SentenceTransformer(DEFAULT_MODEL if model is None else model)
        self.embedding = self.predict  # function alias

    def predict(self, text: str or List, batch_size: int = None):
        self.model.eval()
        text = preprocess(text) if type(text) is str else [preprocess(t) for t in text]
        if batch_size is None:
            batch_size = len(text)
        with torch.no_grad():
            out = self.model.encode(text, batch_size=batch_size)
        return out

    def similarity(self, text_a: str, text_b: str):
        self.model.eval()
        with torch.no_grad():
            vectors = self.model.encode([text_a, text_b])
        return cosine_similarity(vectors[0], vectors[1])
