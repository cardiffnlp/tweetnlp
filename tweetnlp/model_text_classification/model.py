""" Simple interface for CardiffNLP twitter models. """
# TODO: Add preprocessing to handle the twitter username
import logging
import csv
import json
import os
import re
import urllib.request
from typing import List, Dict

import torch
from urlextract import URLExtract
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

URLEx = URLExtract()
DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/tweetnlp/classification"
MODEL_LIST = {
    'emotion': "cardiffnlp/twitter-roberta-base-emotion",
    'emoji': "cardiffnlp/twitter-roberta-base-emoji",
    'hate': "cardiffnlp/twitter-roberta-base-hate",
    'irony': "cardiffnlp/twitter-roberta-base-irony",
    'offensive': "cardiffnlp/twitter-roberta-base-offensive",
    'sentiment': "cardiffnlp/twitter-roberta-base-sentiment-latest",
    'sentiment_multilingual': "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    'topic_single': 'cardiffnlp/tweet-topic-21-single',
    'topic_multi': 'cardiffnlp/tweet-topic-21-multi'
}


def load_model(model, local_files_only: bool = False):
    config = AutoConfig.from_pretrained(model, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model, config=config, local_files_only=local_files_only)
    return config, tokenizer, model


def download_id2label(task):
    path = f'{DEFAULT_CACHE_DIR}/id2label/{task}'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # download label mapping
        mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
        with urllib.request.urlopen(mapping_link) as f:
            html = f.read().decode('utf-8').split("\n")
            csvreader = csv.reader(html, delimiter='\t')
        labels = [row[1] for row in csvreader if len(row) > 1]
        id2label = {str(n): _l for n, _l in enumerate(labels)}
        with open(path, 'w') as f:
            json.dump(id2label, f)
    else:
        with open(path) as f:
            id2label = json.load(f)
    return id2label


def preprocess(text):
    text = re.sub(r"@[A-Z,0-9]+", "@user", text)
    urls = URLEx.find_urls(text)
    for _url in urls:
        try:
            text = text.replace(_url, "http")
        except re.error:
            logging.warning(f're.error:\t - {text}\n\t - {_url}')
    return text



class Classifier:

    def __init__(self, model_name: str, max_length: int, id_to_label: Dict = None, multi_label: bool = False):
        try:
            self.config, self.tokenizer, self.model = load_model(model_name)
        except Exception:
            self.config, self.tokenizer, self.model = load_model(model_name, local_files_only=True)
        self.max_length = max_length
        self.multi_label = multi_label
        if id_to_label is None:
            self.id_to_label = {str(v): k for k, v in self.config.label2id.items()}
        else:
            self.id_to_label = id_to_label
        # GPU setup
        self.device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'
        self.parallel = torch.cuda.device_count() > 1
        if self.parallel:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        logging.debug(f'{torch.cuda.device_count()} GPUs are in use')

    def predict(self, text: str or List, batch_size: int = None):
        self.model.eval()
        single_input_flag = False
        if type(text) is str:
            text = [text]
            single_input_flag = True
        text = [preprocess(t) for t in text]
        if batch_size is None:
            batch_size = len(text)
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        predictions = []
        probs = []
        with torch.no_grad():
            for i in range(len(_index) - 1):
                tmp_text = text[_index[i]: _index[i+1]]
                encoded_input = self.tokenizer.batch_encode_plus(
                    tmp_text,
                    max_length=self.max_length,
                    return_tensors='pt',
                    padding=True,
                    truncation=True)
                output = self.model(**{k: v.to(self.device) for k, v in encoded_input.items()})
                if self.multi_label:
                    prob = torch.sigmoid(output.logits).cpu().tolist()
                    predictions += [[n for n, p in enumerate(_pr) if p > 0.5] for _pr in prob]
                    probs += [{self.id_to_label[str(n)]: p for n, p in enumerate(_pr)} for _pr in prob]

                else:
                    prob = torch.softmax(output.logits, -1).cpu()
                    probs += prob.max(-1)[0].tolist()
                    predictions += prob.argmax(-1).tolist()

            if self.multi_label:
                out = [{'label': [self.id_to_label[str(_p)] for _p in p], 'probability': pr} for pr, p in zip(probs, predictions)]
            else:
                out = [{'label': self.id_to_label[str(p)], 'probability': pr} for pr, p in zip(probs, predictions)]
        if single_input_flag:
            return out[0]
        return out


class TopicClassification:

    def __init__(self, model: str = None, max_length: int = 128, single_class: bool = False):
        if single_class:
            self.model = Classifier(MODEL_LIST['topic_single'] if model is None else model, max_length)
        else:
            self.model = Classifier(MODEL_LIST['topic_multi'] if model is None else model, max_length, multi_label=True)
        self.topic = self.predict = self.model.predict  # function alias


class Sentiment:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['sentiment'] if model is None else model, max_length, download_id2label('sentiment'))
        self.sentiment = self.predict = self.model.predict  # function alias


class SentimentMultilingual:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['sentiment_multilingual'] if model is None else model, max_length, download_id2label('sentiment'))
        self.sentiment = self.predict = self.model.predict  # function alias


class Offensive:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['offensive'] if model is None else model, max_length, download_id2label('offensive'))
        self.offensive = self.predict = self.model.predict  # function alias


class Irony:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['irony'] if model is None else model, max_length, download_id2label('irony'))
        self.irony = self.predict = self.model.predict  # function alias


class Hate:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['hate'] if model is None else model, max_length, download_id2label('hate'))
        self.hate = self.predict = self.model.predict  # function alias


class Emotion:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['emotion'] if model is None else model, max_length, download_id2label('emotion'))
        self.emotion = self.predict = self.model.predict  # function alias


class Emoji:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['emoji'] if model is None else model, max_length, download_id2label('emoji'))
        self.emoji = self.predict = self.model.predict  # function alias
