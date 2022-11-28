import logging

from .text_classification.model import Sentiment, Offensive, Irony, Hate, Emotion, Emoji,\
    TopicClassification, StanceAtheism, StanceAbortion, StanceClimate, StanceHillary, StanceFeminist
from .text_classification.dataset import load_dataset_text_classification

from .ner.model import NER
from .ner.dataset import load_dataset_ner

from .mlm.model import LanguageModel
from .sentence_embedding.model import SentenceEmbedding

TASK_CLASS = {
    'sentiment': [Sentiment, load_dataset_text_classification],
    'offensive': [Offensive, load_dataset_text_classification],
    'irony': [Irony, load_dataset_text_classification],
    'hate': [Hate, load_dataset_text_classification],
    'emotion': [Emotion, load_dataset_text_classification],
    'emoji': [Emoji, load_dataset_text_classification],
    'stance_abortion': [StanceAbortion, load_dataset_text_classification],
    'stance_atheism': [StanceAtheism, load_dataset_text_classification],
    'stance_climate': [StanceClimate, load_dataset_text_classification],
    'stance_feminist': [StanceFeminist, load_dataset_text_classification],
    'stance_hillary': [StanceHillary, load_dataset_text_classification],
    'topic_classification': [TopicClassification, load_dataset_text_classification],
    'ner': [NER, load_dataset_ner],
    'language_model': [LanguageModel],
    'sentence_embedding': [SentenceEmbedding],
}


def load_model(task_type: str = None, model_name: str = None, max_length: int = 128, *args, **kwargs):
    assert task_type is not None or model_name is not None, "task_type or model_name should be specified"
    logging.debug(f'[config]: {task_type}, {model_name}, {max_length}')
    if task_type not in TASK_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_CLASS.keys()}')
    model_loader = TASK_CLASS[task_type][0]
    model_class = model_loader(model_name=model_name, max_length=max_length, *args, **kwargs)
    if not hasattr(model_class, 'predict'):
        raise NotImplementedError(f'Class for {task_type} does not have predict function')
    return model_class


def load_dataset(task_type: str = None, dataset_type: str = None, dataset_name: str = None, *args, **kwargs):
    assert task_type is not None or dataset_type is not None, "task_type or dataset_type should be specified"
    logging.debug(f'[config]: {task_type}, {dataset_type}, {dataset_name}')
    if task_type not in TASK_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_CLASS.keys()}')
    data_loader = TASK_CLASS[task_type][1]
    return data_loader(task_type=task_type, dataset_type=dataset_type, dataset_name=dataset_name, *args, **kwargs)


