# basic loader
from .loader import load_model, load_dataset
# model class
from .ner.model import NER
from .text_classification.model import Sentiment, Offensive, Emoji, Emotion, Irony, Hate

# dataset class
from .text_classification.dataset import load_dataset_text_classification
