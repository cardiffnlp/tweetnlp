""" Simple interface for CardiffNLP masked language models. """
# TODO: Add preprocessing to handle the twitter username
import logging
import re
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-2021-124m"


def load_model(model, local_files_only: bool = False):
    config = AutoConfig.from_pretrained(model, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    model = AutoModelForMaskedLM.from_pretrained(model, config=config, local_files_only=local_files_only)
    return config, tokenizer, model


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class LanguageModel:

    def __init__(self, model: str = None, max_length: int = 128):
        try:
            self.config, self.tokenizer, self.model = load_model(DEFAULT_MODEL if model is None else model)
        except Exception:
            self.config, self.tokenizer, self.model = load_model(DEFAULT_MODEL if model is None else model,
                                                                 local_files_only=True)
        self.max_length = max_length
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')
        self.mask_prediction = self.predict  # function alias

    def ids_to_tokens(self, token_ids):
        tokens = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [re.sub(r'\s+\Z', '', re.sub(r'\A\s+', '', t)) for t in tokens]

    def predict(self,
                text: str or List,
                batch_size: int = None,
                best_n: int = 10,
                worst_n: int = None):
        self.model.eval()
        single_input_flag = False
        if type(text) is str:
            text = [text]
            single_input_flag = True
        for t in text:
            assert t.count(self.tokenizer.mask_token) == 1, f"one <mask> token should be in the texts: {t}"
        text = [preprocess(t) for t in text]
        if batch_size is None:
            batch_size = len(text)
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        predictions = []
        probs = []
        mask_positions = []
        input_ids = []

        with torch.no_grad():
            for i in range(len(_index) - 1):
                tmp_text = text[_index[i]: _index[i+1]]
                encode = self.tokenizer.batch_encode_plus(tmp_text,
                                                          max_length=self.max_length,
                                                          return_tensors='pt',
                                                          padding=True,
                                                          truncation=True)

                mask_positions += [e.index(self.tokenizer.mask_token_id) for e in encode['input_ids'].cpu().tolist()]
                output = self.model(**{k: v.to(self.device) for k, v in encode.items()})
                prob = torch.softmax(output.logits, dim=-1)
                predictions += prob.argsort(-1, descending=True).cpu().tolist()
                probs += prob.cpu().tolist()
                input_ids += encode['input_ids'].cpu().tolist()

        outputs = []
        assert len(predictions) == len(mask_positions)
        for pred, mask_position, prob, input_id in zip(predictions, mask_positions, probs, input_ids):
            tmp_out = {'best_tokens': self.ids_to_tokens(pred[mask_position][:best_n]),
                       'best_scores': prob[mask_position][:best_n],
                       'best_sentences': [self.tokenizer.decode(
                           [pr[p] if n == mask_position else i for n, (pr, i) in enumerate(zip(pred, input_id))],
                           skip_special_tokens=True) for p in range(best_n)]
                       }
            if worst_n is not None:
                tmp_out['worst_tokens'] = self.ids_to_tokens(pred[mask_position][-worst_n:])
                tmp_out['worst_scores'] = prob[mask_position][-worst_n:]
                tmp_out['worst_sentences'] = [self.tokenizer.decode(
                    [pr[-p] if n == mask_position else i for n, (pr, i) in enumerate(zip(pred, input_id))],
                    skip_special_tokens=True) for p in range(worst_n)]
            outputs.append(tmp_out)
        if single_input_flag:
            return outputs[0]
        return outputs
