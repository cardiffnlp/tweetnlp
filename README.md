[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/tweetnlp/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/tweetnlp.svg)](https://badge.fury.io/py/tweetnlp)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)
[![PyPI status](https://img.shields.io/pypi/status/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)

# TweetNLP
TweetNLP for all the NLP enthusiasts working on Twitter! 
The python library `tweetnlp` provides a collection of useful tools to analyze/understand tweets such as sentiment analysis,
emoji prediction, and named-entity recognition, powered by state-of-the-art language modeling trained on tweets.


Resources:
- Quick Tour with Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu?usp=sharing)
- Play with TweetNLP Online Demo: [link](https://tweetnlp.org/demo/)


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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=KAZYjeskBqL4&line=4&uniqifier=1)

The classification module consists of six different tasks (Sentiment Analysis, Irony Detection, Hate Detection, Offensive Detection, Emoji Prediction, and Emotion Analysis).
In each example, the model is instantiated by `tweetnlp.load("task-name")`, and run the prediction by giving a text or a list of 
texts.


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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=WeREiLEjBlrj&line=3&uniqifier=1)

The information extraction module consists of named-entity recognition (NER) model specifically trained for tweets.
The model is instantiated by `tweetnlp.load("ner")`, and run the prediction by giving a text or a list of texts.


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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=COOoZHVAFCIG&line=2&uniqifier=1)

Masked language model predicts masked token in the given sentence. This is instantiated by `tweetnlp.load('language_model')`, and run the prediction by giving a text or a list of texts. Please make sure that each text has `<mask>` token, that is the objective of the model to predict.

```python
model = tweetnlp.load('language_model')  # Or `model = tweetnlp.LanguageModel()` 
model.mask_prediction("How many more <mask> until opening day? 😩")  # Or `model.predict`
```

## Tweet/Sentence Embedding
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KLMaGFLmbXWeM9eWIYgGkRZS0d85RJLu#scrollTo=MUT31bNQYTNz&line=2&uniqifier=1)

Tweet embedding model produces a fixed length embedding for a tweet. The embedding represents the semantics of the tweet, and this can be used a semantic search of tweets by using the similarity in betweein the embeddings. Model is instantiated by `tweet_nlp.load('sentence_embedding')`, and run the prediction by giving a text or a list of texts.

- ***Get Embedding***

```python
model = tweetnlp.load('sentence_embedding')  # Or `model = tweetnlp.SentenceEmbedding()` 

# Get sentence embedding
tweet = "I will never understand the decision making of the people of Alabama. Their new Senator is a definite downgrade. You have served with honor.  Well done."
vectors = model.predict(tweet)
vectors.shape
>>> (768,)

# Get sentence embedding (multiple inputs)
tweet_corpus = [
    "Free, fair elections are the lifeblood of our democracy. Charges of unfairness are serious. But calling an election unfair does not make it so. Charges require specific allegations and then proof. We have neither here.",
    "Trump appointed judge Stephanos Bibas ",
    "If your members can go to Puerto Rico they can get their asses back in the classroom. @CTULocal1",
    "@PolitiBunny @CTULocal1 Political leverage, science said schools could reopen, teachers and unions protested to keep'em closed and made demands for higher wages and benefits, they're usin Covid as a crutch at the expense of life and education.",
    "Congratulations to all the exporters on achieving record exports in Dec 2020 with a growth of 18 % over the previous year. Well done &amp; keep up this trend. A major pillar of our govt's economic policy is export enhancement &amp; we will provide full support to promote export culture.",
    "@ImranKhanPTI Pakistan seems a worst country in term of exporting facilities. I am a small business man and if I have to export a t-shirt having worth of $5 to USA or Europe. Postal cost will be around $30. How can we grow as an exporting country if this situation prevails. Think about it. #PM",
    "The thing that doesn’t sit right with me about “nothing good happened in 2020” is that it ignores the largest protest movement in our history. The beautiful, powerful Black Lives Matter uprising reached every corner of the country and should be central to our look back at 2020.",
    "@JoshuaPotash I kinda said that in the 2020 look back for @washingtonpost",
    "Is this a confirmation from Q that Lin is leaking declassified intelligence to the public? I believe so. If @realDonaldTrump didn’t approve of what @LLinWood is doing he would have let us know a lonnnnnng time ago. I’ve always wondered why Lin’s Twitter handle started with “LLin” https://t.co/0G7zClOmi2",
    "@ice_qued @realDonaldTrump @LLinWood Yeah 100%",
    "Tomorrow is my last day as Senator from Alabama.  I believe our opportunities are boundless when we find common ground. As we swear in a new Congress &amp; a new President, demand from them that they do just that &amp; build a stronger, more just society.  It’s been an honor to serve you." 
    "The mask cult can’t ever admit masks don’t work because their ideology is based on feeling like a “good person”  Wearing a mask makes them a “good person” &amp; anyone who disagrees w/them isn’t  They can’t tolerate any idea that makes them feel like their self-importance is unearned",
    "@ianmSC Beyond that, they put such huge confidence in masks so early with no strong evidence that they have any meaningful benefit, they don’t want to backtrack or admit they were wrong. They put the cart before the horse, now desperate to find any results that match their hypothesis.",
]
vectors = model.predict(tweet_corpus, batch_size=3)
vectors.shape
>>> (12, 768)
```

- ***Similarity Search***

```python
sims = []
for n, i in enumerate(tweet_corpus):
  _sim = model.similarity(tweet, i)
  sims.append([n, _sim])
print(f'anchor tweet: {tweet}\n')
for m, (n, s) in enumerate(sorted(sims, key=lambda x: x[1], reverse=True)[:3]):
  print(f' - top {m}: {tweet_corpus[n]}\n - similaty: {s}\n')

>>> anchor tweet: I will never understand the decision making of the people of Alabama. Their new Senator is a definite downgrade. You have served with honor.  Well done.
>>> 
>>>  - top 0: Is this a confirmation from Q that Lin is leaking declassified intelligence to the public? I believe so. If @realDonaldTrump didn’t approve of what @LLinWood is doing he would have let us know a lonnnnnng time ago. I’ve always wondered why Lin’s Twitter handle started with “LLin” https://t.co/0G7zClOmi2
>>>  - similaty: 1.0787510714776494
>>> 
>>>  - top 1: Tomorrow is my last day as Senator from Alabama.  I believe our opportunities are boundless when we find common ground. As we swear in a new Congress &amp; a new President, demand from them that they do just that &amp; build a stronger, more just society.  It’s been an honor to serve you.The mask cult can’t ever admit masks don’t work because their ideology is based on feeling like a “good person”  Wearing a mask makes them a “good person” &amp; anyone who disagrees w/them isn’t  They can’t tolerate any idea that makes them feel like their self-importance is unearned
>>>  - similaty: 1.0151820570776409
>>> 
>>>  - top 2: @ice_qued @realDonaldTrump @LLinWood Yeah 100%
>>>  - similaty: 1.0036063366758512
```

## Reference (TBA)
- TweetEval
- T-NER
- TimeLM
- etc