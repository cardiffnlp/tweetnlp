[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/tweetnlp/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/tweetnlp.svg)](https://badge.fury.io/py/tweetnlp)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)
[![PyPI status](https://img.shields.io/pypi/status/tweetnlp.svg)](https://pypi.python.org/pypi/tweetnlp/)

# TweetNLP
TweetNLP for all the NLP enthusiasts working on Twitter! 
The python library `tweetnlp` provides a collection of useful tools to analyze/understand tweets such as sentiment analysis,
emoji prediction, and named-entity recognition, powered by state-of-the-art language modeling trained on tweets.

***News (September 2022):*** Our paper presenting TweetNLP, "TweetNLP: Cutting-Edge Natural Language Processing for Social Media", has been accepted as an EMNLP 2022 system demonstration!! Camera-ready version can be found [here](https://arxiv.org/abs/2206.14774).


Resources:
- Quick Tour with Colab Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp?usp=sharing)
- Play with the TweetNLP Online Demo: [link](https://tweetnlp.org/demo/)
- EMNLP 2022 paper: [link](https://arxiv.org/abs/2206.14774)

Table of Contents:
1. [***Load Model & Dataset***](https://github.com/cardiffnlp/tweetnlp/tree/add_training#model--dataset)
2. [***Fine-tune Model***](https://github.com/cardiffnlp/tweetnlp/tree/add_training#model-fine-tuning)

## Get Started

Install TweetNLP via pip on your console. 
```shell
pip install tweetnlp
```
## Model & Dataset

In this section, you will learn how to get the models and datasets with `tweetnlp`.
The models follow [huggingface model](https://huggingface.co/) and the datasets are in the format of [huggingface datasets](https://huggingface.co/docs/datasets/load_hub).
Easy introductions of huggingface models and datasets should be found at [huggingface webpage](https://huggingface.co/), so
please check them if you are new to huggingface.

### Tweet Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=KAZYjeskBqL4)

The classification module consists of six different tasks (Topic Classification, Sentiment Analysis, Irony Detection, Hate Speech Detection, Offensive Language Detection, Emoji Prediction, and Emotion Analysis).
In each example, the model is instantiated by `tweetnlp.load_model("task-name")`, and run the prediction by passing a text or a list of texts as argument to the corresponding function.

- ***Topic Classification***: The aim of this task is, given a tweet to assign topics related to its content. The task is formed as a supervised multi-label classification problem where each tweet is assigned one or more topics from a total of 19 available topics. The topics were carefully curated based on Twitter trends with the aim to be broad and general and consist of classes such as: arts and culture, music, or sports. Our internally-annotated dataset contains over 10K manually-labeled tweets (check the paper [here](https://arxiv.org/abs/2209.09824), or the [huggingface dataset page](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single)).

```python
import tweetnlp

# MULTI-LABEL MODEL 
model = tweetnlp.load_model('topic_classification')  # Or `model = tweetnlp.TopicClassification()`
model.topic("Jacob Collier is a Grammy-awarded English artist from London.")  # Or `model.predict`
>>> {'label': ['celebrity_&_pop_culture', 'music']}
# Note: the probability of the multi-label model is the output of sigmoid function on binary prediction whether each topic is positive or negative.
model.topic("Jacob Collier is a Grammy-awarded English artist from London.", return_probability=True)
>>> {'label': ['celebrity_&_pop_culture', 'music'],
 'probability': {'arts_&_culture': 0.037371691316366196,
  'business_&_entrepreneurs': 0.010188567452132702,
  'celebrity_&_pop_culture': 0.92448890209198,
  'diaries_&_daily_life': 0.03425711765885353,
  'family': 0.00796138122677803,
  'fashion_&_style': 0.020642118528485298,
  'film_tv_&_video': 0.08062587678432465,
  'fitness_&_health': 0.006343095097690821,
  'food_&_dining': 0.0042883665300905704,
  'gaming': 0.004327300935983658,
  'learning_&_educational': 0.010652057826519012,
  'music': 0.8291937112808228,
  'news_&_social_concern': 0.24688217043876648,
  'other_hobbies': 0.020671198144555092,
  'relationships': 0.020371075719594955,
  'science_&_technology': 0.0170074962079525,
  'sports': 0.014291072264313698,
  'travel_&_adventure': 0.010423899628221989,
  'youth_&_student_life': 0.008605164475739002}}

# SINGLE-LABEL MODEL
model = tweetnlp.load_model('topic_classification', multi_label=False)  # Or `model = tweetnlp.TopicClassification(multi_label=False)`
model.topic("Jacob Collier is a Grammy-awarded English artist from London.")
>>> {'label': 'pop_culture'}
# NOTE: the probability of the sinlge-label model the softmax over the label.
model.topic("Jacob Collier is a Grammy-awarded English artist from London.", return_probability=True)
>>> {'label': 'pop_culture',
 'probability': {'arts_&_culture': 9.20625461731106e-05,
  'business_&_entrepreneurs': 6.916998972883448e-05,
  'pop_culture': 0.9995898604393005,
  'daily_life': 0.00011083036952186376,
  'sports_&_gaming': 8.668467489769682e-05,
  'science_&_technology': 5.152115045348182e-05}}

# GET DATASET
dataset_multi_label, label2id_multi_label = tweetnlp.load_dataset('topic_classification')
dataset_single_label, label2id_single_label = tweetnlp.load_dataset('topic_classification', multi_label=False)
```


- ***Sentiment Analysis***: The sentiment analysis task integrated in TweetNLP is a simplified version where the goal is to predict the sentiment of a tweet with one of the three following labels: positive, neutral or negative. The base dataset for English is the unified TweetEval version of the Semeval-2017 dataset from the task on Sentiment Analysis in Twitter (check the paper [here](https://arxiv.org/pdf/2010.12421.pdf)).

```python
import tweetnlp

# ENGLISH MODEL
model = tweetnlp.load_model('sentiment')  # Or `model = tweetnlp.Sentiment()` 
model.sentiment("Yes, including Medicare and social security saving👍")  # Or `model.predict`
>>> {'label': 'positive'}
model.sentiment("Yes, including Medicare and social security saving👍", return_probability=True)
>>> {'label': 'positive', 'probability': {'negative': 0.004584966693073511, 'neutral': 0.19360853731632233, 'positive': 0.8018065094947815}}

# MULTILINGUAL MODEL
model = tweetnlp.load_model('sentiment', multilingual=True)  # Or `model = tweetnlp.Sentiment(multilingual=True)` 
model.sentiment("天気が良いとやっぱり気持ち良いなあ✨")
>>> {'label': 'positive'}
model.sentiment("天気が良いとやっぱり気持ち良いなあ✨", return_probability=True)
>>> {'label': 'positive', 'probability': {'negative': 0.028369612991809845, 'neutral': 0.08128828555345535, 'positive': 0.8903420567512512}}

# GET DATASET (ENGLISH)
dataset, label2id = tweetnlp.load_dataset('sentiment')
# GET DATASET (MULTILINGUAL)
for l in ['all', 'arabic', 'english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish']:
    dataset_multilingual, label2id_multilingual = tweetnlp.load_dataset('sentiment', multilingual=True, task_language=l)
```

- ***Irony Detection***: This is a binary classification task where given a tweet, the goal is to detect whether it is ironic or not. It is based on the Irony Detection dataset from the SemEval 2018 task (check the paper [here](https://arxiv.org/pdf/2010.12421.pdf)).

```python
import tweetnlp

# MODEL
model = tweetnlp.load_model('irony')  # Or `model = tweetnlp.Irony()` 
model.irony('If you wanna look like a badass, have drama on social media')  # Or `model.predict`
>>> {'label': 'irony'}
model.irony('If you wanna look like a badass, have drama on social media', return_probability=True)
>>> {'label': 'irony', 'probability': {'non_irony': 0.08390884101390839, 'irony': 0.9160911440849304}} 

# GET DATASET
dataset, label2id = tweetnlp.load_dataset('irony')
```

- ***Hate Speech Detection***: The hate speech dataset consists of detecting whether a tweet is hateful towards women or immigrants. It is based on the Detection of Hate Speech task at SemEval 2019 (check the paper [here](https://arxiv.org/pdf/2010.12421.pdf)).

```python
import tweetnlp

# MODEL
model = tweetnlp.load_model('hate')  # Or `model = tweetnlp.Hate()` 
model.hate('Whoever just unfollowed me you a bitch')  # Or `model.predict`
>>> {'label': 'not-hate'}
model.hate('Whoever just unfollowed me you a bitch', return_probability=True)
>>> {'label': 'non-hate', 'probability': {'non-hate': 0.7263831496238708, 'hate': 0.27361682057380676}}

# GET DATASET
dataset, label2id = tweetnlp.load_dataset('hate')
```

- ***Offensive Language Identification***: This task consists in identifying whether some form of offensive language is present in a tweet. For our benchmark we rely on the SemEval2019 OffensEval dataset (check the paper [here](https://arxiv.org/pdf/2010.12421.pdf)).

```python
import tweetnlp

# MODEL
model = tweetnlp.load_model('offensive')  # Or `model = tweetnlp.Offensive()` 
model.offensive("All two of them taste like ass.")  # Or `model.predict`
>>> {'label': 'offensive'}
model.offensive("All two of them taste like ass.", return_probability=True)
>>> {'label': 'offensive', 'probability': {'non-offensive': 0.16420328617095947, 'offensive': 0.8357967734336853}}

# GET DATASET
dataset, label2id = tweetnlp.load_dataset('offensive')
```

- ***Emoji Prediction***: The goal of emoji prediction is to predict the final emoji on a given tweet. The dataset used to fine-tune our models is the TweetEval adaptation from the SemEval 2018 task on Emoji Prediction (check the paper [here](https://arxiv.org/pdf/2010.12421.pdf)), including 20 emoji as labels (❤, 😍, 😂, 💕, 🔥, 😊, 😎, ✨, 💙, 😘, 📷, 🇺🇸, ☀, 💜, 😉, 💯, 😁, 🎄, 📸, 😜).	

```python
import tweetnlp

# MODEL
model = tweetnlp.load_model('emoji')  # Or `model = tweetnlp.Emoji()` 
model.emoji('Beautiful sunset last night from the pontoon @TupperLakeNY')  # Or `model.predict`
>>> {'label': '😊'}
model.emoji('Beautiful sunset last night from the pontoon @TupperLakeNY', return_probability=True)
>>> {'label': '📷',
 'probability': {'❤': 0.13197319209575653,
  '😍': 0.11246423423290253,
  '😂': 0.008415069431066513,
  '💕': 0.04842926934361458,
  '🔥': 0.014528146013617516,
  '😊': 0.1509675830602646,
  '😎': 0.08625403046607971,
  '✨': 0.01616635173559189,
  '💙': 0.07396604865789413,
  '😘': 0.03033279813826084,
  '📷': 0.16525287926197052,
  '🇺🇸': 0.020336611196398735,
  '☀': 0.00799981877207756,
  '💜': 0.016111424192786217,
  '😉': 0.012984540313482285,
  '💯': 0.012557178735733032,
  '😁': 0.031386848539114,
  '🎄': 0.006829539313912392,
  '📸': 0.04188741743564606,
  '😜': 0.011156936176121235}}

# GET DATASET
dataset, label2id = tweetnlp.load_dataset('emoji')
```

- ***Emotion Recognition***: Given a tweet, this task consists of associating it with its most appropriate emotion. As a reference dataset we use the SemEval 2018 task on Affect in Tweets, simplified to only four emotions used in TweetEval: anger, joy, sadness and optimism (check the paper [here](https://arxiv.org/pdf/2010.12421.pdf)).

```python
import tweetnlp

# MODEL
model = tweetnlp.load_model('emotion')  # Or `model = tweetnlp.Emotion()` 
model.emotion('I love swimming for the same reason I love meditating...the feeling of weightlessness.')  # Or `model.predict`
>>> {'label': 'joy'}
model.emotion('I love swimming for the same reason I love meditating...the feeling of weightlessness.', return_probability=True)
>>> {'label': 'optimism', 'probability': {'joy': 0.01367587223649025, 'optimism': 0.7345258593559265, 'anger': 0.1770714670419693, 'sadness': 0.07472680509090424}}

# GET DATASET
dataset, label2id = tweetnlp.load_dataset('emotion')
```

### Named Entity Recognition
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=WeREiLEjBlrj)

This module consists of a named-entity recognition (NER) model specifically trained for tweets. The model is instantiated by `tweetnlp.load_model("ner")`, and runs the prediction by giving a text or a list of texts as argument to the `ner` function (check the paper [here](https://arxiv.org/abs/2210.03797), or the [huggingface dataset page](https://huggingface.co/datasets/tner/tweetner7)). 

```python3
import tweetnlp

# MODEL
model = tweetnlp.load_model('ner')  # Or `model = tweetnlp.NER()` 
model.ner('Jacob Collier is a Grammy-awarded English artist from London.')  # Or `model.predict`
>>> [{'type': 'person', 'entity': 'Jacob Collier'}, {'type': 'event', 'entity': ' Grammy'}, {'type': 'location', 'entity': ' London'}]
# Note: the probability for the predicted entity is the mean of the probabilities over the sub-tokens representing the entity. 
model.ner('Jacob Collier is a Grammy-awarded English artist from London.', return_probability=True)  # Or `model.predict`
>>> [
  {'type': 'person', 'entity': 'Jacob Collier', 'probability': 0.9905318220456442},
  {'type': 'event', 'entity': ' Grammy', 'probability': 0.19164378941059113},
  {'type': 'location', 'entity': ' London', 'probability': 0.9607000350952148}
]

# GET DATASET
dataset, label2id = tweetnlp.load_dataset('ner')
```

### Question Answering
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=reZDePaBmYhA&line=4&uniqifier=1)

This module consists of a question answering model specifically trained for tweets.
The model is instantiated by `tweetnlp.load_model("question_answering")`, 
and runs the prediction by giving a question or a list of questions along with a context or a list of contexts
as argument to the `question_answering` function (check the paper [here](https://arxiv.org/abs/2210.03992), or the [huggingface dataset page](https://huggingface.co/datasets/lmqg/qg_tweetqa)). 

```python3
import tweetnlp

# MODEL
model = tweetnlp.load_model('question_answering')  # Or `model = tweetnlp.QuestionAnswering()` 
model.question_answering(
  question='who created the post as we know it today?',
  context="'So much of The Post is Ben,' Mrs. Graham said in 1994, three years after Bradlee retired as editor. 'He created it as we know it today.'— Ed O'Keefe (@edatpost) October 21, 2014"
)  # Or `model.predict`
>>> {'generated_text': 'ben'}

# GET DATASET
dataset = tweetnlp.load_dataset('question_answering')
```

### Question Answer Generation
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=uqd7sBHhnwym&line=6&uniqifier=1)

This module consists of a question & answer pair generation specifically trained for tweets.
The model is instantiated by `tweetnlp.load_model("question_answer_generation")`, 
and runs the prediction by giving a context or a list of contexts
as argument to the `question_answer_generation` function (check the paper [here](https://arxiv.org/abs/2210.03992), or the [huggingface dataset page](https://huggingface.co/datasets/lmqg/qag_tweetqa)). 

```python3
import tweetnlp

# MODEL
model = tweetnlp.load_model('question_answer_generation')  # Or `model = tweetnlp.QuestionAnswerGeneration()` 
model.question_answer_generation(
  text="'So much of The Post is Ben,' Mrs. Graham said in 1994, three years after Bradlee retired as editor. 'He created it as we know it today.'— Ed O'Keefe (@edatpost) October 21, 2014"
)  # Or `model.predict`
>>> [
    {'question': 'who created the post?', 'answer': 'ben'},
    {'question': 'what did ben do in 1994?', 'answer': 'he retired as editor'}
]

# GET DATASET
dataset = tweetnlp.load_dataset('question_answer_generation')
```

### Language Modeling
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=COOoZHVAFCIG&line=1&uniqifier=1)

The masked language model predicts the masked token in the given sentence. This is instantiated by `tweetnlp.load_model('language_model')`, and runs the prediction by giving a text or a list of texts as argument to the `mask_prediction` function. Please make sure that each text has a `<mask>` token, since that is eventually the following by the objective of the model to predict.

```python
import tweetnlp
model = tweetnlp.load_model('language_model')  # Or `model = tweetnlp.LanguageModel()` 
model.mask_prediction("How many more <mask> until opening day? 😩", best_n=2)  # Or `model.predict`
>>> {'best_tokens': ['days', 'hours'],
 'best_scores': [5.498564104033932e-11, 4.906026140893971e-10],
 'best_sentences': ['How many more days until opening day? 😩',
  'How many more hours until opening day? 😩']}
```

### Tweet Embedding
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=MUT31bNQYTNz)

The tweet embedding model produces a fixed length embedding for a tweet. The embedding represents the semantics by meaning of the tweet, and this can be used for semantic search of tweets by using the similarity between the embeddings. Model is instantiated by `tweet_nlp.load_model('sentence_embedding')`, and run the prediction by passing a text or a list of texts as argument to the `embedding` function.

- ***Get Embedding***

```python
import tweetnlp
model = tweetnlp.load_model('sentence_embedding')  # Or `model = tweetnlp.SentenceEmbedding()` 

# Get sentence embedding
tweet = "I will never understand the decision making of the people of Alabama. Their new Senator is a definite downgrade. You have served with honor.  Well done."
vectors = model.embedding(tweet)
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
vectors = model.embedding(tweet_corpus, batch_size=4)
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

 - top 0: Tomorrow is my last day as Senator from Alabama.  I believe our opportunities are boundless when we find common ground. As we swear in a new Congress &amp; a new President, demand from them that they do just that &amp; build a stronger, more just society.  It’s been an honor to serve you.The mask cult can’t ever admit masks don’t work because their ideology is based on feeling like a “good person”  Wearing a mask makes them a “good person” &amp; anyone who disagrees w/them isn’t  They can’t tolerate any idea that makes them feel like their self-importance is unearned
 - similaty: 0.7480925982953287

 - top 1: Trump appointed judge Stephanos Bibas 
 - similaty: 0.6289173306344258

 - top 2: Free, fair elections are the lifeblood of our democracy. Charges of unfairness are serious. But calling an election unfair does not make it so. Charges require specific allegations and then proof. We have neither here.
 - similaty: 0.6017154109745276
```

### Resources & Custom Model Loading 

Here is a table of the default model used in each task. 

| Task                              | Model                                                                                                                                                   | Dataset |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
|Topic Classification (single-label)| [cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all](https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-single-all) | [cardiffnlp/tweet_topic_single](https://huggingface.co/datasets/cardiffnlp/tweet_topic_single) |
|Topic Classification (multi-label) | [cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all](https://huggingface.co/cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all)   | [cardiffnlp/tweet_topic_multi](https://huggingface.co/datasets/cardiffnlp/tweet_topic_multi) |
|Sentiment Analysis (Multilingual)  | [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)                                   | [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) |
|Sentiment Analysis                 | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)                             | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Irony Detection                    | [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)                                                   | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Hate Detection                     | [cardiffnlp/twitter-roberta-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate)                                                     | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Offensive Detection                | [cardiffnlp/twitter-roberta-base-offensive](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive)                                           | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Emoji Prediction                   | [cardiffnlp/twitter-roberta-base-emoji](https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji)                                                   | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Emotion Analysis                   | [cardiffnlp/twitter-roberta-base-emotion](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)                                               | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Named Entity Recognition           | [tner/roberta-large-tweetner7-all](https://huggingface.co/tner/roberta-large-tweetner7-all)                                                        | [tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7) |
|Question Answering                 | [lmqg/t5-small-tweetqa-qa](https://huggingface.co/lmqg/t5-small-tweetqa-qa)                                                                             | [lmqg/qg_tweetqa](https://huggingface.co/datasets/lmqg/qg_tweetqa) |
|Question Answer Generation         | [lmqg/t5-base-tweetqa-qag](https://huggingface.co/lmqg/t5-base-tweetqa-qag)                                                                             | [lmqg/qag_tweetqa](https://huggingface.co/datasets/lmqg/qag_tweetqa) |
|Language Modeling                  | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m)                                           | TBA |
|Tweet Embedding                    | [cambridgeltl/tweet-roberta-base-embeddings-v1](https://huggingface.co/cambridgeltl/tweet-roberta-base-embeddings-v1)                                   | TBA |


To use an other model from local/huggingface modelhub, one can simply provide the model path/alias to the `load_model` function.
Below is an example to load a model for NER.

```python
import tweetnlp
tweetnlp.load_model('ner', model_name='tner/twitter-roberta-base-2019-90m-tweetner7-continuous')
```

## Model Fine-tuning
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp#scrollTo=2plrPTqk7OHp)

TweetNLP provides an easy interface to fine-tune language models on the datasets supported by HuggingFace for model hosting/fine-tuning with [RAY TUNE](https://docs.ray.io/en/latest/tune/index.html) for parameter search.

- Supported Tasks: `sentiment`, `offensive`, `irony`, `hate`, `emotion`, `topic_classification`

The results of experiments with `tweetnlp`'s trainer can be found in the following table. Results are competitive and can be used as baselines for each task.
See [the leaderboard page](https://github.com/cardiffnlp/tweetnlp/blob/main/FINETUNING_RESULT.md) to know more about the results.

| task      | language_model                                                                                                |   eval_f1 |   eval_f1_macro |   eval_accuracy | link                                                                                                                              |
|:----------|:--------------------------------------------------------------------------------------------------------------|----------:|----------------:|----------------:|:----------------------------------------------------------------------------------------------------------------------------------|
| emoji     | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.46 |            0.35 |            0.46 | [cardiffnlp/twitter-roberta-base-2021-124m-emoji](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-emoji)         |
| emotion   | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.83 |            0.79 |            0.83 | [cardiffnlp/twitter-roberta-base-2021-124m-emotion](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-emotion)     |
| hate      | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.56 |            0.53 |            0.56 | [cardiffnlp/twitter-roberta-base-2021-124m-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-hate)           |
| irony     | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.79 |            0.78 |            0.79 | [cardiffnlp/twitter-roberta-base-2021-124m-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-irony)         |
| offensive | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.86 |            0.82 |            0.86 | [cardiffnlp/twitter-roberta-base-2021-124m-offensive](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-offensive) |
| sentiment | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.71 |            0.72 |            0.71 | [cardiffnlp/twitter-roberta-base-2021-124m-sentiment](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-sentiment) |
| topic_classification (single) | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.9  |  0.8  |            0.9  | [cardiffnlp/twitter-roberta-base-2021-124m-topic-single](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-topic-single)                 |                                                                                                                               
| topic_classification (multi)  | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m) |      0.75 |            0.56 |            0.54 | [cardiffnlp/twitter-roberta-base-2021-124m-topic-multi](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-topic-multi)                   |
| sentiment (multilingual)      | [cardiffnlp/twitter-xlm-roberta-base](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base)             |      0.69 |  0.69 |            0.69 | [cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual)         |                                                                                                                               


### Example 
The following example will reproduce our irony model [cardiffnlp/twitter-roberta-base-2021-124m-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m-irony).

```python
import logging
import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# load dataset
dataset, label_to_id = tweetnlp.load_dataset("irony")
# load trainer class
trainer_class = tweetnlp.load_trainer("irony")
# setup trainer
trainer = trainer_class(
    language_model='cardiffnlp/twitter-roberta-base-2021-124m',  # language model to fine-tune
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_test='test',
    split_train='train',
    split_validation='validation',
    output_dir='model_ckpt/irony' 
)
# start model fine-tuning with parameter optimization
trainer.train(
  eval_step=50,  # each `eval_step`, models are validated on the validation set 
  n_trials=10,  # number of trial at parameter optimization
  search_range_lr=[1e-6, 1e-4],  # define the search space for learning rate (min and max value)
  search_range_epoch=[1, 6],  # define the search space for epoch (min and max value)
  search_list_batch=[4, 8, 16, 32, 64]  # define the search space for batch size (list of integer to test) 
)
# evaluate model on the test set
trainer.evaluate()
>>> {
  "eval_loss": 1.3228046894073486,
  "eval_f1": 0.7959183673469388,
  "eval_f1_macro": 0.791350632069195,
  "eval_accuracy": 0.7959183673469388,
  "eval_runtime": 2.2267,
  "eval_samples_per_second": 352.084,
  "eval_steps_per_second": 44.01
}
# save model locally (saved at `{output_dir}/best_model` as default)
trainer.save_model()
# run prediction
trainer.predict('If you wanna look like a badass, have drama on social media')
>>> {'label': 'irony'}
# push your model on huggingface hub
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias='twitter-roberta-base-2021-124m-irony')
```
The saved checkpoint can be loaded as a custom model as below.
```python
import tweetnlp
model = tweetnlp.load_model('irony', model_name="model_ckpt/irony/best_model")
```
If `split_validation` is not given, trainer will do a single run with default parameters without parameter search.

## Reference Paper

For more details, please read the accompanying [TweetNLP's reference paper](https://arxiv.org/pdf/2206.14774.pdf). If you use TweetNLP in your research, please use the following `bib` entry to cite the reference paper:

```
@inproceedings{camacho-collados-etal-2022-tweetnlp,
    title={{T}weet{NLP}: {C}utting-{E}dge {N}atural {L}anguage {P}rocessing for {S}ocial {M}edia},
    author={Camacho-Collados, Jose and Rezaee, Kiamehr and Riahi, Talayeh and Ushio, Asahi and Loureiro, Daniel and Antypas, Dimosthenis and Boisson, Joanne and Espinosa-Anke, Luis and Liu, Fangyu and Mart{\'\i}nez-C{\'a}mara, Eugenio and others},
    author = "Ushio, Asahi  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2022",
    address = "Abu Dhabi, U.A.E.",
    publisher = "Association for Computational Linguistics",
}
```
