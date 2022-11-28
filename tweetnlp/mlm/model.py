""" Simple interface for CardiffNLP masked language models. """
import logging
import re
from typing import List

import torch
from ..util import load_model

DEFAULT_MLM_MODEL = "cardiffnlp/twitter-roberta-base-2021-124m"


class LanguageModel:

    def __init__(self, model_name: str = None, max_length: int = 128, use_auth_token: bool = False):
        model_name = DEFAULT_MLM_MODEL if model_name is None else model_name
        self.config, self.tokenizer, self.model = load_model(
            model_name, task='masked_language_model', use_auth_token=use_auth_token)
        self.max_length = max_length
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')
        self.mask_prediction = self.predict  # function alias
        self.model.eval()

    def ids_to_tokens(self, token_ids):
        tokens = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        return [re.sub(r'\s+\Z', '', re.sub(r'\A\s+', '', t)) for t in tokens]

    def predict(self,
                text: str or List,
                batch_size: int = None,
                best_n: int = 10,
                worst_n: int = None):

        single_input_flag = type(text) is str
        text = [text] if single_input_flag else text
        assert all(t.count(self.tokenizer.mask_token) == 1 for t in text),\
            f"{self.tokenizer.mask_token} token not found: {text}"
        assert all(type(t) is str for t in text), text
        batch_size = len(text) if batch_size is None else batch_size
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
