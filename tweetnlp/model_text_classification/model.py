""" Simple interface for CardiffNLP twitter models. """
import csv
import json
import os
import urllib.request
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/tweetnlp/classification"
MODEL_LIST = {
    'emotion': "cardiffnlp/twitter-roberta-base-emotion",
    'emoji': "cardiffnlp/twitter-roberta-base-emoji",
    'hate': "cardiffnlp/twitter-roberta-base-hate",
    'irony': "cardiffnlp/twitter-roberta-base-irony",
    'offensive': "cardiffnlp/twitter-roberta-base-offensive",
    'sentiment': "cardiffnlp/twitter-roberta-base-sentiment-latest",
    # 'stance': "cardiffnlp/twitter-roberta-base-stance-climate"
}


def load_model(model, local_files_only: bool = False):
    config = AutoConfig.from_pretrained(model, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model, config=config, local_files_only=local_files_only)
    return config, tokenizer, model


def download_label2dict(task):
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
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


class Classifier:

    def __init__(self, model_name: str, max_length: int, label_to_id: Dict):
        try:
            self.config, self.tokenizer, self.model = load_model(model_name)
        except Exception:
            self.config, self.tokenizer, self.model = load_model(model_name, local_files_only=True)
        self.max_length = max_length
        self.label_to_id = label_to_id

    def predict(self, text: str or List, batch_size: int = None):
        single_input_flag = False
        if type(text) is str:
            text = [text]
            single_input_flag = True
        text = [preprocess(t) for t in text]
        if batch_size is None:
            batch_size = len(text)
        _index = list(range(0, len(text), batch_size)) + [len(text) + 1]
        output_list = []
        for i in range(len(_index) - 1):
            tmp_text = text[_index[i]: _index[i+1]]
            encoded_input = self.tokenizer.batch_encode_plus(tmp_text, max_length=self.max_length, return_tensors='pt')
            output = self.model(**encoded_input)
            pred = output.logits.argmax(-1).cpu().tolist()
            output_list += pred
        if single_input_flag:
            return self.label_to_id[str(output_list[0])]
        else:
            return [self.label_to_id[str(p)] for p in output_list]


class Sentiment:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['sentiment'] if model is None else model, max_length, download_label2dict('sentiment'))
        self.sentiment = self.predict = self.model.predict  # function alias


class Offensive:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['offensive'] if model is None else model, max_length, download_label2dict('offensive'))
        self.offensive = self.predict = self.model.predict  # function alias


class Irony:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['irony'] if model is None else model, max_length, download_label2dict('irony'))
        self.irony = self.predict = self.model.predict  # function alias


class Hate:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['hate'] if model is None else model, max_length, download_label2dict('hate'))
        self.hate = self.predict = self.model.predict  # function alias


class Emotion:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['emotion'] if model is None else model, max_length, download_label2dict('emotion'))
        self.emotion = self.predict = self.model.predict  # function alias


class Emoji:

    def __init__(self, model: str = None, max_length: int = 128):
        self.model = Classifier(MODEL_LIST['emoji'] if model is None else model, max_length, download_label2dict('emoji'))
        self.emoji = self.predict = self.model.predict  # function alias
