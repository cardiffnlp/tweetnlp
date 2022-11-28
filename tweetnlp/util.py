import re
import logging
import urllib.request
from urlextract import URLExtract
from datasets.features import Sequence, ClassLabel
from datasets.dataset_dict import DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoConfig,\
    AutoModelForMaskedLM


def get_preprocessor(processor_type: str = None):
    url_ex = URLExtract()

    if processor_type is None:
        def preprocess(text):
            text = re.sub(r"@[A-Z,0-9]+", "@user", text)
            urls = url_ex.find_urls(text)
            for _url in urls:
                try:
                    text = text.replace(_url, "http")
                except re.error:
                    logging.warning(f're.error:\t - {text}\n\t - {_url}')
            return text

    elif processor_type == 'tweet_topic':

        def preprocess(tweet):
            # mask web urls
            urls = url_ex.find_urls(tweet)
            for url in urls:
                tweet = tweet.replace(url, "{{URL}}")
            # format twitter account
            tweet = re.sub(r"\b(\s*)(@[\S]+)\b", r'\1{\2@}', tweet)
            return tweet
    else:
        raise ValueError(f"unknown type: {processor_type}")

    return preprocess


def get_label2id(dataset: DatasetDict, label_name: str = 'label'):
    label_info = dataset[list(dataset.keys())[0]].features[label_name]
    while True:
        if type(label_info) is Sequence:
            label_info = label_info.feature
        else:
            assert type(label_info) is ClassLabel, f"Error at retrieving label information {label_info}"
            break
    return {k: n for n, k in enumerate(label_info.names)}


def load_model(model: str, task: str = 'sequence_classification', use_auth_token: bool = False):
    try:
        urllib.request.urlopen('http://google.com')
        no_network = False
    except Exception:
        no_network = True
    config = AutoConfig.from_pretrained(model, use_auth_token=use_auth_token, local_files_only=no_network)
    tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=use_auth_token, local_files_only=no_network)
    if task == 'sequence_classification':
        model = AutoModelForSequenceClassification.from_pretrained(
            model, config=config, use_auth_token=use_auth_token, local_files_only=no_network
        )
    elif task == 'token_classification':
        model = AutoModelForTokenClassification.from_pretrained(
            model, config=config, use_auth_token=use_auth_token, local_files_only=no_network
        )
    elif task == 'masked_language_model':
        model = AutoModelForMaskedLM.from_pretrained(
            model, config=config, use_auth_token=use_auth_token, local_files_only=no_network)
    else:
        raise ValueError(f'unknown task: {task}')
    return config, tokenizer, model

