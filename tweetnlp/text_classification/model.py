""" Simple interface for CardiffNLP twitter models. """
import logging
import os
from typing import List, Dict

import torch

from ..util import load_model, get_preprocessor

DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/tweetnlp/classification"
MODEL_LIST = {
    'emotion': {
        "default": "cardiffnlp/twitter-roberta-base-emotion"
    },
    'emoji': {
        "default": "cardiffnlp/twitter-roberta-base-emoji"
    },
    'hate': {
        "default": "cardiffnlp/twitter-roberta-base-hate"
    },
    'irony': {
        "default": "cardiffnlp/twitter-roberta-base-irony"
    },
    'offensive': {
        "default": "cardiffnlp/twitter-roberta-base-offensive"
    },
    'stance_abortion': {
        "default": "cardiffnlp/twitter-roberta-base-stance-abortion"
    },
    'stance_atheism': {
        "default": "cardiffnlp/twitter-roberta-base-stance-atheism"
    },
    'stance_climate': {
        "default": "cardiffnlp/twitter-roberta-base-stance-climate"
    },
    'stance_feminist': {
        "default": "cardiffnlp/twitter-roberta-base-stance-feminist"
    },
    'stance_hillary': {
        "default": "cardiffnlp/twitter-roberta-base-stance-hillary"
    },
    'sentiment': {
        "default": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    },
    'topic_classification': {
        "single_label": 'cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all',
        "multi_label": 'cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all'
    }
}


class Classifier:

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 multi_label: bool = False,
                 use_auth_token: bool = False,
                 loaded_model_config_tokenizer: Dict = None):
        if loaded_model_config_tokenizer is not None:
            assert all(i in loaded_model_config_tokenizer.keys() for i in ['model', 'config', 'tokenizer'])
            self.config = loaded_model_config_tokenizer['config']
            self.tokenizer = loaded_model_config_tokenizer['tokenizer']
            self.model = loaded_model_config_tokenizer['model']
        else:
            assert model_name is not None, "model_name is required"
            logging.debug(f'loading {model_name}')
            self.config, self.tokenizer, self.model = load_model(
                model_name, task='sequence_classification', use_auth_token=use_auth_token)
        self.max_length = max_length
        self.multi_label = multi_label
        self.id_to_label = {str(v): k for k, v in self.config.label2id.items()}
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')

        self.model.eval()
        self.preprocess = get_preprocessor()

    def predict(self,
                text: str or List,
                batch_size: int = None,
                return_probability: bool = False,
                skip_preprocess: bool = False):
        single_input_flag = type(text) is str
        text = [text] if single_input_flag else text
        if not skip_preprocess:
            text = [self.preprocess(t) for t in text]
        assert all(type(t) is str for t in text), text
        batch_size = len(text) if batch_size is None else batch_size
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        probs = []
        with torch.no_grad():
            for i in range(len(_index) - 1):
                encoded_input = self.tokenizer.batch_encode_plus(
                    text[_index[i]: _index[i+1]],
                    max_length=self.max_length,
                    return_tensors='pt',
                    padding=True,
                    truncation=True)
                output = self.model(**{k: v.to(self.device) for k, v in encoded_input.items()})
                if self.multi_label:
                    probs += torch.sigmoid(output.logits).cpu().tolist()
                else:
                    probs += torch.softmax(output.logits, -1).cpu().tolist()

        if return_probability:
            if self.multi_label:
                out = [{
                    'label': [self.id_to_label[str(n)] for n, p in enumerate(_pr) if p > 0.5],
                    'probability': {self.id_to_label[str(n)]: p for n, p in enumerate(_pr)}
                } for _pr in probs]
            else:
                out = [{
                    'label': self.id_to_label[str(p.index(max(p)))],
                    'probability': {self.id_to_label[str(n)]: _p for n, _p in enumerate(p)}
                } for p in probs]
        else:
            if self.multi_label:
                out = [{'label': [self.id_to_label[str(n)] for n, p in enumerate(_pr) if p > 0.5]} for _pr in probs]
            else:
                out = [{'label': self.id_to_label[str(p.index(max(p)))]} for p in probs]
        if single_input_flag:
            return out[0]
        return out


class TopicClassification(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 multi_label: bool = True,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['topic_classification']['multi_label' if multi_label else 'single_label']
        super().__init__(model_name, max_length=max_length, multi_label=multi_label, use_auth_token=use_auth_token)
        self.topic = self.predict
        self.preprocess = get_preprocessor('tweet_topic')


class Sentiment(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 multilingual: bool = False,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['sentiment']['multilingual' if multilingual else 'default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.sentiment = self.predict


class Offensive(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['offensive']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.offensive = self.predict


class Irony(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['irony']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.irony = self.predict


class Hate(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['hate']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.hate = self.predict


class Emotion(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['emotion']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.emotion = self.predict


class Emoji(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['emoji']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.emoji = self.predict


class StanceAbortion(Classifier):

    def __init__(self,
                 model_name: str = None,
                 max_length: int = 128,
                 use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['stance_abortion']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.stance = self.predict


class StanceAtheism(Classifier):

    def __init__(self, model_name: str = None, max_length: int = 128):
        if model_name is None:
            model_name = MODEL_LIST['stance_atheism']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.stance = self.predict


class StanceClimate(Classifier):

    def __init__(self, model_name: str = None, max_length: int = 128, use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['stance_climate']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.stance = self.predict


class StanceFeminist(Classifier):

    def __init__(self, model_name: str = None, max_length: int = 128, use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['stance_feminist']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.stance = self.predict


class StanceHillary(Classifier):

    def __init__(self, model_name: str = None, max_length: int = 128, use_auth_token: bool = False):
        if model_name is None:
            model_name = MODEL_LIST['stance_hillary']['default']
        super().__init__(model_name, max_length=max_length, use_auth_token=use_auth_token)
        self.stance = self.predict

