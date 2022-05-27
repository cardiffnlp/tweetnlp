[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/tweetnlp/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/tweetnlp.svg)](https://badge.fury.io/py/tweetnlp)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)
[![PyPI status](https://img.shields.io/pypi/status/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)

# TweetNLP
All you need to understand Twitter is `tweetnlp`!

## Get Started

Install TweetNLP with 
```shell
pip install tweetnlp
```

and get started with 

```python
import tweetnlp
```


## Tweet/Sentence Classification

- ***Sentiment Analysis***

```python
model = tweetnlp.load('sentiment')  # Or `model = tweetnlp.Sentiment()` 
model.sentiment("How many more days until opening day? 😩")  # Or `model.predict`
```

- ***Irony Detection***

```python
model = tweetnlp.load('irony')  # Or `model = tweetnlp.Irony()` 
model.irony('If you wanna look like a badass, have drama on social media')  # Or `model.predict`
```

- ***Hate Detection***

```python
model = tweetnlp.load('hate')  # Or `model = tweetnlp.Hate()` 
model.hate('Whoever just unfollowed me you a bitch')  # Or `model.predict`
```

- ***Offensive Detection***

```python
model = tweetnlp.load('offensive')  # Or `model = tweetnlp.Offensive()` 
model.offensive("All two of them taste like ass. ")  # Or `model.predict`
```

- ***Emoji Prediction***

```python
model = tweetnlp.load('emoji')  # Or `model = tweetnlp.Emoji()` 
model.emoji('Beautiful sunset last night from the pontoon @ Tupper Lake, New York')  # Or `model.predict`
```

- ***Emotion Analysis***

```python
model = tweetnlp.load('emotion')  # Or `model = tweetnlp.Emotion()` 
model.emotion('I love swimming for the same reason I love meditating...the feeling of weightlessness.')  # Or `model.predict`
```

## Information Extraction

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

- ***Sentence Embedding***

```python
model = tweetnlp.load('sentence_embedding')  # Or `model = tweetnlp.SentenceEmbedding()` 
model.embedding("How many more days until opening day? 😩")  # Or `model.predict`
```

