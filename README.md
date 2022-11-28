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


## Get Started

Install TweetNLP via pip on your console. 
```shell
pip install tweetnlp
```
## Model & Dataset

### Tweet Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp?usp=sharing)

The classification module consists of seven different tasks (Topic Classification, Sentiment Analysis, Irony Detection, 
Hate Detection, Offensive Detection, Emoji Prediction, and Emotion Analysis). In each example, the model is instantiated 
by `tweetnlp.load_model("task-name")`, and run the prediction by giving a text or a list of texts.

- ***Topic Classification***: This model classifies given tweet into 19 categories. As default, it returns all relevant topics to the tweet, 
  so the output could be a list of topics. Single-label model (return single topic instead) can be also loaded by  
  `tweetnlp.load_model('topic_classification', multi_label=False)` that classifies a tweet into 6 major topics. Check the [paper](https://arxiv.org/abs/2209.09824) for more detail.

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


- ***Sentiment Analysis***: Binary classification of `positive`/`negative`. This module supports 8 different languages now 
  (Arabic/English/French/Spanish/German/Portuguese/Hindi/Italian).

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
for l in ['arabic', 'english', 'french', 'german', 'hindi', 'italian', 'portuguese', 'spanish']:
    dataset_multilingual, label2id_multilingual = tweetnlp.load_dataset('sentiment', multilingual=True, task_language=l)
```

- ***Irony Detection***: Binary classification of whether the tweet is irony or not.

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

- ***Hate Speech Detection***: Binary classification of whether the tweet is hate or not.

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

- ***Offensive Language Identification***: Binary classification of whether the tweet is offensive or not.

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

- ***Emoji Prediction***: Predict appropriate single emoji to the tweet from 20 emojis (❤, 😍, 😂, 💕, 🔥, 😊, 😎, ✨, 💙, 😘, 📷, 🇺🇸, ☀, 💜, 😉, 💯, 😁, 🎄, 📸, 😜).	

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

- ***Emotion Recognition***: Predict the emotion of the tweet from four classes: `anger`/`joy`/`optimism`/`sadness`.

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

### Information Extraction
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp?usp=sharing)

The information extraction module consists of named-entity recognition (NER) model specifically trained for tweets.
The model is instantiated by `tweetnlp.load_model("ner")`, and run the prediction by giving a text or a list of texts.


- ***Named Entity Recognition***

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

### Language Modeling
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp?usp=sharing)

Masked language model predicts masked token in the given sentence. This is instantiated by `tweetnlp.load_model('language_model')`, and run the prediction by giving a text or a list of texts. Please make sure that each text has `<mask>` token, that is the objective of the model to predict.

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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/104MtF9MXkDFimlJLr4SFBX0HjidLTfvp?usp=sharing)

Tweet embedding model produces a fixed length embedding for a tweet. The embedding represents the semantics of the tweet, and this can be used a semantic search of tweets by using the similarity in betweein the embeddings. Model is instantiated by `tweet_nlp.load('sentence_embedding')`, and run the prediction by giving a text or a list of texts.

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
|Named Entity Recognition           | [tner/roberta-large-tweetner7-all](https://huggingface.co/tner/tner/roberta-large-tweetner7-all)                                                        | [tner/tweetner7](https://huggingface.co/datasets/tner/tweetner7) |
|Sentiment Analysis (Multilingual)  | [cardiffnlp/twitter-xlm-roberta-base-sentiment](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)                                   | [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) |
|Sentiment Analysis                 | [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)                             | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Irony Detection                    | [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony)                                                   | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Hate Detection                     | [cardiffnlp/twitter-roberta-base-hate](https://huggingface.co/cardiffnlp/twitter-roberta-base-hate)                                                     | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Offensive Detection                | [cardiffnlp/twitter-roberta-base-offensive](https://huggingface.co/cardiffnlp/twitter-roberta-base-offensive)                                           | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Emoji Prediction                   | [cardiffnlp/twitter-roberta-base-emoji](https://huggingface.co/cardiffnlp/twitter-roberta-base-emoji)                                                   | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Emotion Analysis                   | [cardiffnlp/twitter-roberta-base-emotion](https://huggingface.co/cardiffnlp/twitter-roberta-base-emotion)                                               | [tweet_eval](https://huggingface.co/datasets/tweet_eval) |
|Language Modeling                  | [cardiffnlp/twitter-roberta-base-2021-124m](https://huggingface.co/cardiffnlp/twitter-roberta-base-2021-124m)                                           | TBA |
|Tweet Embedding                    | [cambridgeltl/tweet-roberta-base-embeddings-v1](https://huggingface.co/cambridgeltl/tweet-roberta-base-embeddings-v1)                                   | TBA |


To use other model from local/huggingface modelhub, one can simply provide model path/alias at the model loading.

```python
tweetnlp.load_model('ner', model_name='tner/twitter-roberta-base-2019-90m-tweetner7-continuous')
```

## Model Fine-tuning
```python
import logging
import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

dataset, label_to_id = tweetnlp.load_dataset("hate")
trainer_class = tweetnlp.load_trainer("hate")
trainer = trainer_class(
    language_model='cardiffnlp/twitter-roberta-base-dec2021',
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_test='test',
    split_train='train',
    split_validation='validation',
    output_dir='model_ckpt/hate'
)
trainer.train(eval_step=50, n_trials=5)
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias='twitter-roberta-base-dec2021-hate')
```

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
