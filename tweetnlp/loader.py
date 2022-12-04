from .text_classification.model import Sentiment, Offensive, Irony, Hate, Emotion, Emoji,\
    TopicClassification, StanceAtheism, StanceAbortion, StanceClimate, StanceHillary, StanceFeminist
from .text_classification.dataset import load_dataset_text_classification
from .text_classification.trainer import TrainerTextClassification

from .ner.model import NER
from .ner.dataset import load_dataset_ner

from .question_answering.model import QuestionAnswering
from .question_answering.dataset import load_dataset_question_answering

from .question_answer_generation.model import QuestionAnswerGeneration
from .question_answer_generation.dataset import load_dataset_question_answer_generation


from .mlm.model import LanguageModel
from .sentence_embedding.model import SentenceEmbedding


TASK_CLASS = {
    'sentiment': [Sentiment, load_dataset_text_classification, TrainerTextClassification],
    'offensive': [Offensive, load_dataset_text_classification, TrainerTextClassification],
    'irony': [Irony, load_dataset_text_classification, TrainerTextClassification],
    'hate': [Hate, load_dataset_text_classification, TrainerTextClassification],
    'emotion': [Emotion, load_dataset_text_classification, TrainerTextClassification],
    'emoji': [Emoji, load_dataset_text_classification, TrainerTextClassification],
    'stance_abortion': [StanceAbortion, load_dataset_text_classification, TrainerTextClassification],
    'stance_atheism': [StanceAtheism, load_dataset_text_classification, TrainerTextClassification],
    'stance_climate': [StanceClimate, load_dataset_text_classification, TrainerTextClassification],
    'stance_feminist': [StanceFeminist, load_dataset_text_classification, TrainerTextClassification],
    'stance_hillary': [StanceHillary, load_dataset_text_classification, TrainerTextClassification],
    'topic_classification': [TopicClassification, load_dataset_text_classification, TrainerTextClassification],
    'ner': [NER, load_dataset_ner, None],
    'language_model': [LanguageModel, None, None],
    'sentence_embedding': [SentenceEmbedding, None, None],
    'question_answering': [QuestionAnswering, load_dataset_question_answering, None],
    'question_answer_generation': [QuestionAnswerGeneration, load_dataset_question_answer_generation, None]
}


def load_model(task_type: str, model_name: str = None, max_length: int = 128, *args, **kwargs):
    if task_type not in TASK_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_CLASS.keys()}')
    model_loader = TASK_CLASS[task_type][0]
    model_class = model_loader(model_name=model_name, max_length=max_length, *args, **kwargs)
    if not hasattr(model_class, 'predict'):
        raise NotImplementedError(f'Class for {task_type} does not have predict function')
    return model_class


def load_dataset(task_type: str, dataset_type: str = None, dataset_name: str = None, *args, **kwargs):
    if task_type not in TASK_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_CLASS.keys()}')
    data_loader = TASK_CLASS[task_type][1]
    if data_loader is None:
        raise NotImplementedError(f"dataset loader is not implemented yet for {task_type}")
    return data_loader(task_type=task_type, dataset_type=dataset_type, dataset_name=dataset_name, *args, **kwargs)


def load_trainer(task_type: str):
    if task_type not in TASK_CLASS:
        raise ValueError(f'UNDEFINED TASK ERROR: {task_type} is not valid, choose from {TASK_CLASS.keys()}')
    trainer = TASK_CLASS[task_type][2]
    if trainer is None:
        raise NotImplementedError(f"trainer is not implemented yet for {task_type}")
    return trainer
