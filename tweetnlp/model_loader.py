import logging

from .model_ner.model import NER
from .model_text_classification.model import Sentiment, Offensive, Irony, Hate, Emotion, Emoji

TASK_MODEL_CLASS = {
    'ner': NER,
    'sentiment': Sentiment,
    'offensive': Offensive,
    'irony': Irony,
    'hate': Hate,
    'emotion': Emotion,
    'emoji': Emoji
}


def load(task_type: str = 'ner', model: str = None, max_length: int = 128):
    logging.debug(f'loading model instance: {task_type}, {model}, {max_length}')
    if task_type not in TASK_MODEL_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_MODEL_CLASS.keys()}')
    model_class = TASK_MODEL_CLASS[task_type](model=model, max_length=max_length)
    if not hasattr(model_class, 'predict'):
        raise NotImplementedError(f'Class for {task_type} does not have predict function')
    return model_class

