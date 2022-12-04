import logging
from typing import List
from statistics import mean
import torch

from .allennlp_crf import ConditionalRandomField, allowed_transitions
from ..util import load_model, get_preprocessor


DEFAULT_MODEL = "tner/roberta-large-tweetner7-all"


class NER:

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        model_name = DEFAULT_MODEL if model_name is None else model_name
        logging.debug(f'loading {model_name}')
        self.config, self.tokenizer, self.model = load_model(
            model_name, task='token_classification', use_auth_token=use_auth_token)
        self.max_length = max_length
        self.id_to_label = {v: k for k, v in self.config.label2id.items()}
        # load crf layer
        self.crf_layer = None
        if 'crf_state_dict' in self.config.to_dict().keys():
            logging.debug('use CRF')
            self.crf_layer = ConditionalRandomField(
                num_tags=len(self.model.config.id2label),
                constraints=allowed_transitions(constraint_type="BIO", labels=self.model.config.id2label)
            )
            self.crf_layer.load_state_dict(
                {k: torch.FloatTensor(v) for k, v in self.model.config.crf_state_dict.items()}
            )

        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
            if self.crf_layer is not None:
                self.crf_layer = torch.nn.DataParallel(self.crf_layer)
        self.model.to(self.device)
        if self.crf_layer is not None:
            self.crf_layer.to(self.device)
        self.model.eval()
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')

        self.preprocess = get_preprocessor()
        self.ner = self.predict

    def predict(self,
                text: str or List,
                batch_size: int = None,
                return_probability: bool = False,
                return_position: bool = False,
                skip_preprocess: bool = False):
        single_input_flag = type(text) is str
        text = [text] if single_input_flag else text
        if not skip_preprocess:
            text = [self.preprocess(i) for i in text]
        assert all(type(t) is str for t in text), text
        batch_size = len(text) if batch_size is None else batch_size
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        probs = []
        preds = []
        inputs = []
        with torch.no_grad():
            for i in range(len(_index) - 1):
                encoded_input = self.tokenizer.batch_encode_plus(
                    text[_index[i]: _index[i + 1]],
                    max_length=self.max_length,
                    return_tensors='pt',
                    padding=True,
                    truncation=True)
                inputs += encoded_input['input_ids'].cpu().detach().int().tolist()
                output = self.model(**{k: v.to(self.device) for k, v in encoded_input.items()})
                prob = torch.softmax(output['logits'], dim=-1).cpu().detach().float().tolist()
                if self.crf_layer is not None:
                    if self.parallel:
                        pred = [tag_seq for tag_seq, _ in self.crf_layer.module.viterbi_tags(output['logits'])]
                    else:
                        pred = [tag_seq for tag_seq, _ in self.crf_layer.viterbi_tags(output['logits'])]
                else:
                    pred = torch.max(prob, dim=-1)[1].cpu().detach().int().tolist()
                probs += [[prob[n][_n][_p] for _n, _p in enumerate(p)] for n, p in enumerate(pred)]
                preds += [[self.id_to_label[_p] for _p in p] for p in pred]

        output = [self.decode_ner_tags(p, prob, i, return_probability, return_position)
                  for p, prob, i in zip(preds, probs, inputs)]
        if single_input_flag:
            return output[0]
        return output

    def decode_ner_tags(self,
                        tag_sequence: List,
                        probability_sequence: List,
                        input_sequence: List,
                        return_probability: bool,
                        return_position: bool):

        def update_collection(_tmp_entity, _tmp_entity_type, _tmp_prob, _tmp_pos, _out):
            if len(_tmp_entity) != 0 and _tmp_entity_type is not None:
                _tmp_data = {'type': _tmp_entity_type, 'entity': self.tokenizer.decode(_tmp_entity)}
                if return_probability:
                    _tmp_data['probability'] = mean(_tmp_prob)
                if return_position:
                    _tmp_data['position'] = _tmp_pos
                    _tmp_data['tokenized_input'] = self.tokenizer.convert_ids_to_tokens(input_sequence)
                _out.append(_tmp_data)
                _tmp_entity = []
                _tmp_prob = []
                _tmp_entity_type = None
            return _tmp_entity, _tmp_entity_type, _tmp_prob, _tmp_pos, _out

        assert len(tag_sequence) == len(input_sequence) == len(probability_sequence), str(
            [len(tag_sequence), len(input_sequence), len(probability_sequence)])
        out = []
        tmp_entity = []
        tmp_prob = []
        tmp_pos = []
        tmp_entity_type = None
        for n, (_l, _i, _prob) in enumerate(zip(tag_sequence, input_sequence, probability_sequence)):
            if _l.startswith('B-'):
                _, _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
                tmp_entity_type = '-'.join(_l.split('-')[1:])
                tmp_entity = [_i]
                tmp_prob = [_prob]
                tmp_pos = [n]
            elif _l.startswith('I-'):
                tmp_tmp_entity_type = '-'.join(_l.split('-')[1:])
                if len(tmp_entity) == 0:
                    # if 'I' not start with 'B', skip it
                    tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out = update_collection(
                        tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
                elif tmp_tmp_entity_type != tmp_entity_type:
                    # if the type does not match with the B, skip
                    tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out = update_collection(
                        tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
                else:
                    tmp_entity.append(_i)
                    tmp_pos.append(n)
                    tmp_prob.append(_prob)
            elif _l == 'O':
                tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out = update_collection(
                    tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
            else:
                raise ValueError(f'unknown tag: {_l}')
        _, _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
        return out
