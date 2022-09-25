# TODO: Add preprocessing to handle the twitter username

import json
import logging
import pickle
import os
import re
from itertools import groupby
from typing import List, Dict
from packaging.version import parse
# from tqdm import tqdm


import torch
from torch import nn
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer
# import numpy as np
# from scipy.stats import bootstrap
# from seqeval.metrics import f1_score, precision_score, recall_score, classification_report

from allennlp import __version__
from allennlp.modules import ConditionalRandomField
if parse("2.10.0") < parse(__version__):
    from allennlp.modules.conditional_random_field import allowed_transitions
else:
    from allennlp.modules.conditional_random_field.conditional_random_field import allowed_transitions


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message

__all__ = 'NER'
PAD_TOKEN_LABEL_ID = nn.CrossEntropyLoss().ignore_index
DEFAULT_MODEL = "tner/twitter-roberta-base-dec2021-tweetner7-2020-2021-continuous"


def user_name_handler(_string):
    pass


def pickle_save(obj, path: str):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def pickle_load(path: str):
    with open(path, "rb") as fp:  # Unpickling
        return pickle.load(fp)


# def f1_with_ci(label_list,
#                pred_list,
#                random_seed: int = 0,
#                n_resamples: int = 1000,
#                confidence_level: List = None,
#                return_ci: bool = False,
#                average='macro'):
#     """ F1 with bootstrap CI (data.shape == (n_sample, 2)) """
#     data = np.array(list(zip(pred_list, label_list)), dtype=object)
#
#     def get_f1(xy, axis=None):
#         assert len(xy.shape) in [2, 3], xy.shape
#         prediction = xy[0]
#         label = xy[1]
#         if axis == -1 and len(xy.shape) == 3:
#             assert average is not None
#             tmp = []
#             for i in tqdm(list(range(len(label)))):
#                 tmp.append(f1_score(label[i, :], prediction[i, :], average=average))
#             return np.array(tmp)
#             # return np.array([f1_score(label[i, :], prediction[i, :], average=average)
#             #                  for i in range(len(label))])
#         assert average is not None
#         return f1_score(label, prediction, average=average)
#
#     confidence_level = confidence_level if confidence_level is not None else [90, 95]
#     mean_score = get_f1(data.T)
#     ci = {}
#     if return_ci:
#         for c in confidence_level:
#             logging.debug(f'computing confidence interval: {c}')
#             res = bootstrap((data,),
#                             get_f1,
#                             confidence_level=c * 0.01,
#                             method='percentile',
#                             n_resamples=n_resamples,
#                             random_state=np.random.default_rng(random_seed))
#             ci[str(c)] = [res.confidence_interval.low, res.confidence_interval.high]
#     return mean_score, ci


# def span_f1(pred_list: List,
#             label_list: List,
#             label2id: Dict,
#             span_detection_mode: bool = False,
#             return_ci: bool = False):
#
#     if span_detection_mode:
#         return_ci = False
#
#         def convert_to_binary_mask(entity_label):
#             if entity_label == 'O':
#                 return entity_label
#             prefix = entity_label.split('-')[0]  # B or I
#             return '{}-entity'.format(prefix)
#
#         label_list = [[convert_to_binary_mask(_i) for _i in i] for i in label_list]
#         pred_list = [[convert_to_binary_mask(_i) for _i in i] for i in pred_list]
#
#     # compute metrics
#     logging.debug('\n{}'.format(classification_report(label_list, pred_list)))
#     m_micro, ci_micro = f1_with_ci(label_list, pred_list, average='micro', return_ci=return_ci)
#     m_macro, ci_macro = f1_with_ci(label_list, pred_list, average='macro', return_ci=return_ci)
#     metric = {
#         "micro/f1": m_micro,
#         "micro/f1_ci": ci_micro,
#         "micro/recall": recall_score(label_list, pred_list, average='micro'),
#         "micro/precision": precision_score(label_list, pred_list, average='micro'),
#         "macro/f1": m_macro,
#         "macro/f1_ci": ci_macro,
#         "macro/recall": recall_score(label_list, pred_list, average='macro'),
#         "macro/precision": precision_score(label_list, pred_list, average='macro'),
#     }
#     target_names = sorted([k.replace('B-', '') for k in label2id.keys() if k.startswith('B-')])
#     if not span_detection_mode:
#         metric["per_entity_metric"] = {}
#         for t in target_names:
#             _label_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in label_list]
#             _pred_list = [[_i if _i.endswith(t) else 'O' for _i in i] for i in pred_list]
#             m, ci = f1_with_ci(_label_list, _pred_list, return_ci=return_ci)
#             metric["per_entity_metric"][t] = {
#                 "f1": m,
#                 "f1_ci": ci,
#                 "precision": precision_score(_label_list, _pred_list),
#                 "recall": recall_score(_label_list, _pred_list)}
#     return metric


def decode_ner_tags(tag_sequence, input_sequence, probability_sequence=None):
    def update_collection(_tmp_entity, _tmp_entity_type, _tmp_prob, _tmp_pos, _out):
        if len(_tmp_entity) != 0 and _tmp_entity_type is not None:
            if _tmp_prob is None:
                _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity, 'position': _tmp_pos})
            else:
                _out.append({'type': _tmp_entity_type, 'entity': _tmp_entity, 'position': _tmp_pos,
                             'probability': _tmp_prob})
            _tmp_entity = []
            _tmp_prob = []
            _tmp_entity_type = None
        return _tmp_entity, _tmp_entity_type, _tmp_prob, _tmp_pos, _out

    probability_sequence = [None] * len(tag_sequence) if probability_sequence is None else probability_sequence
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
            raise ValueError('unknown tag: {}'.format(_l))
    _, _, _, _, out = update_collection(tmp_entity, tmp_entity_type, tmp_prob, tmp_pos, out)
    return out


class NERTokenizer:
    """ NER specific transform pipeline"""

    def __init__(self, transformer_tokenizer: str, id2label: Dict = None, padding_id: int = None):
        """ NER specific transform pipeline """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer, local_files_only=True)
        self.id2label = id2label
        self.padding_id = padding_id if padding_id is not None else PAD_TOKEN_LABEL_ID
        self.label2id = {v: k for k, v in id2label.items()}
        self.pad_ids = {"labels": self.padding_id, "input_ids": self.tokenizer.pad_token_id, "__default__": 0}
        self.prefix = self.__sp_token_prefix()
        self.sp_token_start, _, self.sp_token_end = self.additional_special_tokens(self.tokenizer)

    def __sp_token_prefix(self):
        sentence_go_around = ''.join(self.tokenizer.tokenize('get tokenizer specific prefix'))
        prefix = sentence_go_around[:list(re.finditer('get', sentence_go_around))[0].span()[0]]
        return prefix if prefix != '' else None

    def encode_plus(self, tokens, labels: List = None, max_length: int = 128, mask_by_padding_token: bool = False):
        """ encoder for languages which split words by half-space (mask_by_padding_token should be True for eval) """
        encode = self.tokenizer.encode_plus(
            ' '.join(tokens), max_length=max_length, padding='max_length', truncation=True)
        if labels:
            assert len(tokens) == len(labels)
            fixed_labels = []
            for n, (label, word) in enumerate(zip(labels, tokens)):
                fixed_labels.append(label)
                if n != 0 and self.prefix is None:
                    sub_length = len(self.tokenizer.tokenize(' ' + word))
                else:
                    sub_length = len(self.tokenizer.tokenize(word))
                if sub_length > 1:
                    if mask_by_padding_token:
                        fixed_labels += [PAD_TOKEN_LABEL_ID] * (sub_length - 1)
                    else:
                        if self.id2label[label] == 'O':
                            fixed_labels += [self.label2id['O']] * (sub_length - 1)
                        else:
                            entity = '-'.join(self.id2label[label].split('-')[1:])
                            fixed_labels += [self.label2id['I-{}'.format(entity)]] * (sub_length - 1)
            tmp_padding = PAD_TOKEN_LABEL_ID if mask_by_padding_token else self.pad_ids['labels']
            fixed_labels = [tmp_padding] * len(self.sp_token_start['input_ids']) + fixed_labels
            fixed_labels = fixed_labels[:min(len(fixed_labels), max_length - len(self.sp_token_end['input_ids']))]
            fixed_labels = fixed_labels + [tmp_padding] * (max_length - len(fixed_labels))
            encode['labels'] = fixed_labels
        return encode

    def encode_plus_all(self, tokens: List, labels: List = None, max_length: int = None, mask_by_padding_token: bool = False):
        max_length = self.tokenizer.max_len_single_sentence if max_length is None else max_length
        shared_param = {'max_length': max_length, 'mask_by_padding_token': mask_by_padding_token}
        if labels:
            return [self.encode_plus(*i, **shared_param) for i in zip(tokens, labels)]
        else:
            return [self.encode_plus(i, **shared_param) for i in tokens]

    @staticmethod
    def additional_special_tokens(tokenizer):
        """ get additional special token for beginning/separate/ending, {'input_ids': [0], 'attention_mask': [1]} """
        encode_first = tokenizer.encode_plus('sent1', 'sent2')
        # group by block boolean
        sp_token_mask = [i in tokenizer.all_special_ids for i in encode_first['input_ids']]
        group = [list(g) for _, g in groupby(sp_token_mask)]
        length = [len(g) for g in group]
        group_length = [[sum(length[:n]), sum(length[:n]) + len(g)] for n, g in enumerate(group) if all(g)]
        assert len(group_length) == 3, 'more than 3 special tokens group: {}'.format(group)
        sp_token_start = {k: v[group_length[0][0]:group_length[0][1]] for k, v in encode_first.items()}
        sp_token_sep = {k: v[group_length[1][0]:group_length[1][1]] for k, v in encode_first.items()}
        sp_token_end = {k: v[group_length[2][0]:group_length[2][1]] for k, v in encode_first.items()}
        return sp_token_start, sp_token_sep, sp_token_end


class Dataset(torch.utils.data.Dataset):
    """ torch.utils.data.Dataset wrapper converting into tensor """
    float_tensors = ['attention_mask', 'input_feature']

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def to_tensor(self, name, data):
        if name in self.float_tensors:
            return torch.tensor(data, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.long)

    def __getitem__(self, idx):
        return {k: self.to_tensor(k, v) for k, v in self.data[idx].items()}


def load_hf(model_name, local_files_only: bool = False, label_to_id: Dict = None):
    if label_to_id is not None:
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=len(label_to_id),
            id2label={v: k for k, v in label_to_id.items()},
            label2id=label_to_id,
            local_files_only=local_files_only)
    else:
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
    return AutoModelForTokenClassification.from_pretrained(
        model_name, config=config, local_files_only=local_files_only)


class NER:

    def __init__(self,
                 model: str = None,
                 max_length: int = 128,
                 crf: bool = False,
                 label_to_id: Dict = None):
        self.model_name = model if model is not None else DEFAULT_MODEL
        self.max_length = max_length
        self.crf_layer = None
        # load model
        logging.debug('initialize language model with `{}`'.format(model))
        try:
            self.model = load_hf(self.model_name, label_to_id=label_to_id)
        except Exception:
            self.model = load_hf(self.model_name, True, label_to_id=label_to_id)

        # load crf layer
        if 'crf_state_dict' in self.model.config.to_dict().keys() or crf:
            logging.debug('use CRF')
            self.crf_layer = ConditionalRandomField(
                num_tags=len(self.model.config.id2label),
                constraints=allowed_transitions(constraint_type="BIO", labels=self.model.config.id2label)
            )
            if 'crf_state_dict' in self.model.config.to_dict().keys():
                logging.debug('loading pre-trained CRF layer')
                self.crf_layer.load_state_dict(
                    {k: torch.FloatTensor(v) for k, v in self.model.config.crf_state_dict.items()}
                )
        self.label2id = self.model.config.label2id
        self.id2label = self.model.config.id2label

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
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')

        # load pre processor
        if self.crf_layer is not None:
            self.tokenizer = NERTokenizer(self.model_name, id2label=self.id2label, padding_id=self.label2id['O'])
        else:
            self.tokenizer = NERTokenizer(self.model_name, id2label=self.id2label)

        self.ner = self.predict  # function alias

    def encode_to_loss(self, encode: Dict):
        assert 'labels' in encode
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        if self.crf_layer is not None:
            loss = - self.crf_layer(output['logits'], encode['labels'], encode['attention_mask'])
        else:
            loss = output['loss']
        return loss.mean() if self.parallel else loss

    def encode_to_prediction(self, encode: Dict):
        encode = {k: v.to(self.device) for k, v in encode.items()}
        output = self.model(**encode)
        prob = torch.softmax(output['logits'], dim=-1)
        prob, ind = torch.max(prob, dim=-1)
        prob = prob.cpu().detach().float().tolist()
        ind = ind.cpu().detach().int().tolist()
        if self.crf_layer is not None:
            if self.parallel:
                best_path = self.crf_layer.module.viterbi_tags(output['logits'])
            else:
                best_path = self.crf_layer.viterbi_tags(output['logits'])
            pred_results = []
            for tag_seq, _ in best_path:
                pred_results.append(tag_seq)
            ind = pred_results
        return ind, prob

    def get_data_loader(self,
                        inputs,  # list of tokenized sentences
                        labels: List = None,
                        batch_size: int = None,
                        shuffle: bool = False,
                        drop_last: bool = False,
                        mask_by_padding_token: bool = False,
                        cache_file_feature: str = None,
                        return_loader: bool = True):
        """ Transform features (produced by BERTClassifier.preprocess method) to data loader. """
        if cache_file_feature is not None and os.path.exists(cache_file_feature):
            logging.debug('loading preprocessed feature from {}'.format(cache_file_feature))
            out = pickle_load(cache_file_feature)
        else:
            out = self.tokenizer.encode_plus_all(
                tokens=inputs, labels=labels, max_length=self.max_length, mask_by_padding_token=mask_by_padding_token)

            # remove overflow text
            logging.debug('encode all the data: {}'.format(len(out)))

            # cache the encoded data
            if cache_file_feature is not None:
                os.makedirs(os.path.dirname(cache_file_feature), exist_ok=True)
                pickle_save(out, cache_file_feature)
                logging.debug('preprocessed feature is saved at {}'.format(cache_file_feature))
        if return_loader:
            return torch.utils.data.DataLoader(
                Dataset(out), batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=drop_last)

        return list(Dataset(out))

    # def span_f1(self,
    #             inputs: List,
    #             labels: List,
    #             batch_size: int = None,
    #             cache_file_feature: str = None,
    #             cache_file_prediction: str = None,
    #             span_detection_mode: bool = False,
    #             return_ci: bool = False):
    #     output = self.predict(
    #         inputs=inputs,
    #         labels=labels,
    #         batch_size=batch_size,
    #         cache_file_prediction=cache_file_prediction,
    #         cache_file_feature=cache_file_feature,
    #         return_loader=True
    #     )
    #     return span_f1(output['prediction'], output['label'], self.label2id, span_detection_mode, return_ci=return_ci)

    def predict(self,
                inputs: List or str,
                labels: List = None,
                batch_size: int = None,
                cache_file_feature: str = None,
                cache_file_prediction: str = None,
                return_loader: bool = False):
        single_input_flag = False
        if type(inputs) is str:
            inputs = [inputs]
            single_input_flag = True
        inputs = [i.split(' ') if type(i) is not list else i for i in inputs]
        dummy_label = False
        if labels is None:
            labels = [[0] * len(i) for i in inputs]
            dummy_label = True
        if cache_file_prediction is not None and os.path.exists(cache_file_prediction):
            with open(cache_file_prediction) as f:
                tmp = [json.loads(i) for i in f.read().split('\n') if len(i) > 0]
                pred_list = [i['prediction'] for i in tmp]
                prob_list = [i['probability'] for i in tmp]
            label_list = [[self.id2label[__l] for __l in _l] for _l in labels]
            inputs_list = inputs
        else:
            self.model.eval()
            loader = self.get_data_loader(inputs,
                                          labels=labels,
                                          batch_size=batch_size,
                                          mask_by_padding_token=True,
                                          cache_file_feature=cache_file_feature,
                                          return_loader=return_loader)
            label_list = []
            pred_list = []
            prob_list = []
            ind = 0

            inputs_list = []
            with torch.no_grad():
                for i in loader:
                    if not return_loader:
                        i = {k: torch.unsqueeze(v, 0) for k, v in i.items()}
                    label = i.pop('labels').cpu().tolist()
                    pred, prob = self.encode_to_prediction(i)
                    assert len(label) == len(pred) == len(prob), str([len(label), len(pred), len(prob)])
                    input_ids = i.pop('input_ids').cpu().tolist()
                    for _i, _p, _prob, _l in zip(input_ids, pred, prob, label):
                        assert len(_i) == len(_p) == len(_l)
                        tmp = [(__p, __l, __prob) for __p, __l, __prob in zip(_p, _l, _prob) if __l != PAD_TOKEN_LABEL_ID]
                        tmp_pred = list(list(zip(*tmp))[0])
                        tmp_label = list(list(zip(*tmp))[1])
                        tmp_prob = list(list(zip(*tmp))[2])
                        if len(tmp_label) != len(labels[ind]):
                            if len(tmp_label) < len(labels[ind]):
                                logging.debug('found sequence possibly more than max_length')
                                logging.debug('{}: \n\t - model loader: {}\n\t - label: {}'.format(ind, tmp_label, labels[ind]))
                                tmp_pred = tmp_pred + [self.label2id['O']] * (len(labels[ind]) - len(tmp_label))
                                tmp_prob = tmp_prob + [0.0] * (len(labels[ind]) - len(tmp_label))
                            else:
                                raise ValueError(
                                    '{}: \n\t - model loader: {}\n\t - label: {}'.format(ind, tmp_label, labels[ind]))
                        assert len(tmp_pred) == len(labels[ind])
                        assert len(inputs[ind]) == len(tmp_pred)
                        pred_list.append(tmp_pred)
                        label_list.append(labels[ind])
                        inputs_list.append(inputs[ind])
                        prob_list.append(tmp_prob)
                        ind += 1
            label_list = [[self.id2label[__l] for __l in _l] for _l in label_list]
            pred_list = [[self.id2label[__p] for __p in _p] for _p in pred_list]
            if cache_file_prediction is not None:
                os.makedirs(os.path.dirname(cache_file_prediction), exist_ok=True)
                with open(cache_file_prediction, 'w') as f:
                    for _pred, _prob in zip(pred_list, prob_list):
                        f.write(json.dumps({'prediction': _pred, 'probability': _prob}) + '\n')

        output = {'prediction': pred_list,
                  'probability': prob_list,
                  'input': inputs_list,
                  'entity_prediction': [decode_ner_tags(_p, _i, _prob) for _p, _prob, _i in
                                        zip(pred_list, prob_list, inputs_list)]}
        if not dummy_label:
            output['label'] = label_list
            output['entity_label'] = [decode_ner_tags(_p, _i) for _p, _i in zip(label_list, inputs_list)]
        if single_input_flag:
            for k in output.keys():
                output[k] = output[k][0]
        return output

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def save(self, save_dir):

        def model_state(model):
            if self.parallel:
                return model.module
            return model

        if self.crf_layer is not None:
            model_state(self.model).config.update(
                {'crf_state_dict': {k: v.tolist() for k, v in model_state(self.crf_layer).state_dict().items()}})
        model_state(self.model).save_pretrained(save_dir)
        self.tokenizer.tokenizer.save_pretrained(save_dir)

    # def evaluate(self,
    #              batch_size,
    #              data_split,
    #              cache_file_feature: str = None,
    #              cache_file_prediction: str = None,
    #              span_detection_mode: bool = False,
    #              return_ci: bool = False):
    #     self.eval()
    #     data = get_dataset(data_split)
    #     return self.span_f1(
    #         inputs=data['data'],
    #         labels=data['label'],
    #         batch_size=batch_size,
    #         cache_file_feature=cache_file_feature,
    #         cache_file_prediction=cache_file_prediction,
    #         span_detection_mode=span_detection_mode,
    #         return_ci=return_ci)
