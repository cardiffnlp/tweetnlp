""" Simple interface for CardiffNLP twitter models. """
import logging
from typing import List

import torch
from transformers import pipeline


# DEFAULT_MODEL = "lmqg/t5-small-tweetqa-qag"
DEFAULT_MODEL = "lmqg/t5-base-tweetqa-qag"


class QuestionAnswerGeneration:

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
        self.question_answer_generation = self.predict

    def predict(self,
                text: str or List = None,
                batch_size: int = None,
                add_prefix: bool = True):
        single_input_flag = type(text) is str
        text = [text] if single_input_flag else text
        batch_size = len(text) if batch_size is None else batch_size
        if add_prefix:
            text = [f"generate question and answer: {t}" for t in text]
        output = self.pipe(text, max_length=self.max_length, batch_size=batch_size)
        output = [self.decode_output(o['generated_text']) for o in output]
        if single_input_flag:
            return output[0]
        return output

    @staticmethod
    def decode_output(text):
        entries = [i for i in text.split(" | ") if 'question: ' in i and ', answer: ' in i]
        qa_pairs = [i.replace('question: ', '').split(", answer: ") for i in entries]
        qa_pairs = [{"question": i[0], "answer": i[1]} for i in qa_pairs if len(i) == 2]
        return qa_pairs


if __name__ == '__main__':
    _model = QuestionAnswerGeneration()
    _output = _model.predict(
        "'So much of The Post is Ben,' Mrs. Graham said in 1994, three years after Bradlee retired as editor. 'He created it as we know it today.'— Ed O'Keefe (@edatpost) October 21, 2014",
    )
    print(_output)
    _output = _model.predict(
        [
            "'So much of The Post is Ben,' Mrs. Graham said in 1994, three years after Bradlee retired as editor. 'He created it as we know it today.'— Ed O'Keefe (@edatpost) October 21, 2014",
            "Heresy is any provocative belief or theory that is strongly at variance with established beliefs or customs. A heretic is a proponent of such claims or beliefs. Heresy is distinct from both apostasy, which is the explicit renunciation of one's religion, principles or cause, and blasphemy, which is an impious utterance or action concerning God or sacred things."
        ],
        batch_size=1
    )
    print(_output)


