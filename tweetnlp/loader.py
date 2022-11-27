import logging

from .ner.model import NER
from .text_classification.model import Sentiment, Offensive, Irony, Hate, Emotion, Emoji,\
    TopicClassification
from .mlm.model import LanguageModel
from .sentence_embedding.model import SentenceEmbedding

TASK_MODEL_CLASS = {
    'ner': NER,
    'sentiment': Sentiment,
    'offensive': Offensive,
    'irony': Irony,
    'hate': Hate,
    'emotion': Emotion,
    'emoji': Emoji,
    'topic_classification': TopicClassification,
    'language_model': LanguageModel,
    'sentence_embedding': SentenceEmbedding
}


def load_model(task_type: str = 'topic_classification', model: str = None, max_length: int = 128, *args, **kwargs):
    logging.debug(f'loading model instance: {task_type}, {model}, {max_length}')
    if task_type not in TASK_MODEL_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_MODEL_CLASS.keys()}')
    model_class = TASK_MODEL_CLASS[task_type](model=model, max_length=max_length, *args, **kwargs)
    if not hasattr(model_class, 'predict'):
        raise NotImplementedError(f'Class for {task_type} does not have predict function')
    return model_class

