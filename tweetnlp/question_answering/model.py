""" Simple interface for CardiffNLP twitter models. """
import logging
from typing import List

import torch
from transformers import pipeline


DEFAULT_MODEL = "lmqg/t5-small-tweetqa-qa"


class QuestionAnswering:

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        self.model_name = DEFAULT_MODEL if model_name is None else model_name
        self.num_gpu = torch.cuda.device_count()
        self.num_gpu = -1 if self.num_gpu == 0 else self.num_gpu
        logging.debug(f'{self.num_gpu} GPUs are in use')
        self.pipe = pipeline("text2text-generation", self.model_name, device=self.num_gpu, use_auth_token=use_auth_token)
        self.max_length = max_length
        self.question_answering = self.predict

    def predict(self,
                question: str or List,
                context: str or List = None,
                batch_size: int = None):
        single_input_flag = type(question) is str
        if context is not None:
            assert type(context) is type(question)
            context = [context] if single_input_flag else context
        question = [question] if single_input_flag else question
        assert len(question) == len(context), f"{len(question)} != {len(context)}"
        batch_size = len(question) if batch_size is None else batch_size
        input_text = [f"question: {q}, context: {c}" for q, c in zip(question, context)]
        output = self.pipe(input_text, max_length=self.max_length, batch_size=batch_size)
        if single_input_flag:
            return output[0]
        return output
