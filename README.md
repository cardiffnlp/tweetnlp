[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/tweetnlp/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/tweetnlp.svg)](https://badge.fury.io/py/tweetnlp)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)
[![PyPI status](https://img.shields.io/pypi/status/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)

# TweetNLP
TweetNLP (`tweetnlp`) for all the NLP enthusiasts working on Twitter! 
The python library `tweetnlp` provides a collection of useful tools to analyze/understand tweets such as sentiment analysis,
emoji prediction, and named-entity recognition.

Resources:
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu?usp=sharing)
- [TweetNLP online demo](https://tweetnlp.org/demo/)


## Get Started

Install TweetNLP with 
```shell
pip install tweetnlp
```
and get started with 

```python3
import tweetnlp
```

## Tweet/Sentence Classification
The classification module consists of six different tasks (Sentiment Analysis, Irony Detection, Hate Detection, Offensive Detection, Emoji Prediction, and Emotion Analysis).
In each example, the model is instantiated by `tweetnlp.load("task-name")`, and run the prediction by giving a text or a list of 
texts.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=KAZYjeskBqL4&line=4&uniqifier=1)


- ***Sentiment Analysis***

```python
model = tweetnlp.load('sentiment')  # Or `model = tweetnlp.Sentiment()` 
model.sentiment("Yes, including Medicare and social security saving👍")  # Or `model.predict`
>>> {'label': 'positive', 'probability': 0.8018065094947815}
```

- ***Irony Detection***

```python
model = tweetnlp.load('irony')  # Or `model = tweetnlp.Irony()` 
model.irony('If you wanna look like a badass, have drama on social media')  # Or `model.predict`
>>> {'label': 'irony', 'probability': 0.9160911440849304}
```

- ***Hate Detection***

```python
model = tweetnlp.load('hate')  # Or `model = tweetnlp.Hate()` 
model.hate('Whoever just unfollowed me you a bitch')  # Or `model.predict`
>>> {'label': 'not-hate', 'probability': 0.7263831496238708}
```

- ***Offensive Detection***

```python
model = tweetnlp.load('offensive')  # Or `model = tweetnlp.Offensive()` 
model.offensive("All two of them taste like ass. ")  # Or `model.predict`
>>> {'label': 'offensive', 'probability': 0.8600459098815918}
```

- ***Emoji Prediction***

```python
model = tweetnlp.load('emoji')  # Or `model = tweetnlp.Emoji()` 
model.emoji('Beautiful sunset last night from the pontoon @TupperLakeNY')  # Or `model.predict`
>>> {'label': '😊', 'probability': 0.3179638981819153}
```

- ***Emotion Analysis***

```python
model = tweetnlp.load('emotion')  # Or `model = tweetnlp.Emotion()` 
model.emotion('I love swimming for the same reason I love meditating...the feeling of weightlessness.')  # Or `model.predict`
>>> {'label': 'joy', 'probability': 0.7345258593559265}
```

## Information Extraction
The information extraction module consists of named-entity recognition (NER) model specifically trained for tweets.
The model is instantiated by `tweetnlp.load("ner")`, and run the prediction by giving a text or a list of texts.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=WeREiLEjBlrj&line=3&uniqifier=1)

- ***Named Entity Recognition***

```python3
model = tweetnlp.load('ner')  # Or `model = tweetnlp.NER()` 
model.ner('Jacob Collier is a Grammy-awarded English artist from London.')  # Or `model.predict`
>>> {
    'prediction': ['B-person', 'I-person', 'O', 'O', 'O', 'O', 'O', 'O', 'B-location'],
    'probability': [0.9606876969337463, 0.9834017753601074, 0.9816871285438538, 0.9896021485328674, 0.44137904047966003, 0.375810831785202, 0.8757674694061279, 0.9786785244941711, 0.9398059248924255],
    'input': ['Jacob', 'Collier', 'is', 'a', 'Grammy-awarded', 'English', 'artist', 'from', 'London.'],
    'entity_prediction': [
        {'type': 'person', 'entity': ['Jacob', 'Collier'], 'position': [0, 1], 'probability': [0.9606876969337463, 0.9834017753601074]},
        {'type': 'location', 'entity': ['London.'], 'position': [8], 'probability': [0.9398059248924255]}]
}
```

## Language Modeling
Masked language model predicts masked token in the given sentence. This is instantiated by `tweetnlp.load('language_model')`, and run the prediction by giving a text or a list of texts. Please make sure that each text has `<mask>` token, that is the objective of the model to predict.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=COOoZHVAFCIG&line=2&uniqifier=1)

```python
model = tweetnlp.load('language_model')  # Or `model = tweetnlp.LanguageModel()` 
model.mask_prediction("How many more <mask> until opening day? 😩")  # Or `model.predict`
```

## Tweet/Sentence Embedding

```python
model = tweetnlp.load('sentence_embedding')  # Or `model = tweetnlp.SentenceEmbedding()` 
model.embedding("How many more days until opening day? 😩")  # Or `model.predict`
```

## Reference
- TweetEval
- T-NER
- TimeLM
- etc