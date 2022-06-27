""" Sentence embedding """
# TODO: Add dense search with sentence-transformers
# TODO: Add preprocessing to handle the twitter username
import logging
import re
from typing import List

import torch
from numpy import dot
from numpy.linalg import norm
from urlextract import URLExtract
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "cambridgeltl/tweet-roberta-base-embeddings-v1"
URLEx = URLExtract()


def cosine_similarity(v_a, v_b, eps: float = 1e-5):
    return dot(v_a, v_b) / (norm(v_b) * norm(v_b) + eps)


def preprocess(text):
    text = re.sub(r"@[A-Z,0-9]+", "@user", text)
    urls = URLEx.find_urls(text)
    for _url in urls:
        try:
            text = text.replace(_url, "http")
        except re.error:
            logging.warning(f're.error:\t - {text}\n\t - {_url}')
    return text


class SentenceEmbedding:

    def __init__(self, model: str = None, **kwargs):
        self.model = SentenceTransformer(DEFAULT_MODEL if model is None else model)
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        if torch.cuda.device_count() > 1:
            logging.warning('sentence embedding is not optimized to be used on parallel GPUs')
        # self.parallel = torch.cuda.device_count() > 1
        # if self.parallel:
        #     self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
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
