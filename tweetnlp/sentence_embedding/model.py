""" Sentence embedding """
import logging
from typing import List

import torch
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from ..util import get_preprocessor

DEFAULT_SENTENCE_MODEL = "cambridgeltl/tweet-roberta-base-embeddings-v1"


def cosine_similarity(v_a, v_b, eps: float = 1e-5):
    return dot(v_a, v_b) / (norm(v_b) * norm(v_b) + eps)


class SentenceEmbedding:

    def __init__(self, model_name: str = None, use_auth_token: bool = False, **kwargs):
        model_name = DEFAULT_SENTENCE_MODEL if model_name is None else model_name
        self.model = SentenceTransformer(model_name, use_auth_token=use_auth_token)
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        if torch.cuda.device_count() > 1:
            logging.warning('sentence embedding is not optimized to be used on parallel GPUs')
        self.model.to(self.device)
        self.embedding = self.predict
        self.preprocess = get_preprocessor()
        self.model.eval()

    def predict(self,
                text: str or List,
                batch_size: int = None,
                skip_preprocess: bool = True):
        single_input_flag = type(text) is str
        text = [text] if single_input_flag else text
        assert all(type(t) is str for t in text), text
        if not skip_preprocess:
            text = [self.preprocess(i) for i in text]
        batch_size = len(text) if batch_size is None else batch_size
        with torch.no_grad():
            output = self.model.encode(text, batch_size=batch_size)
        if single_input_flag:
            return output[0]
        return output

    def similarity(self, text_a: str, text_b: str):
        with torch.no_grad():
            vectors = self.model.encode([text_a, text_b])
        return cosine_similarity(vectors[0], vectors[1])
