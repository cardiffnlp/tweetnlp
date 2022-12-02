import os
import logging
import json
import requests
import shutil
import pandas as pd
import tweetnlp
from pprint import pprint

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
sample = [
    "How many more days until opening day? 😩"
    "All two of them taste like ass.",
    "If you wanna look like a badass, have drama on social media",
    "Whoever just unfollowed me you a bitch",
    "I love swimming for the same reason I love meditating...the feeling of weightlessness.",
    "Beautiful sunset last night from the pontoon @ Tupper Lake, New York",
    'Jacob Collier is a Grammy-awarded English artist from London.'
]
lms = ['cardiffnlp/twitter-roberta-base-dec2021', 'cardiffnlp/twitter-roberta-base-2021-124m', 'roberta-base']

##############################
# basic classification tasks #
##############################
for language_model in lms:
    for task in ["irony", "offensive", "emoji", "emotion", "sentiment", "hate"]:
        model_alias = f"{os.path.basename(language_model)}-{task}"
        dataset, label_to_id = tweetnlp.load_dataset(task)
        trainer_class = tweetnlp.load_trainer(task)

        # setup trainer
        trainer = trainer_class(
            language_model=language_model,
            dataset=dataset,
            label_to_id=label_to_id,
            max_length=128,
            split_test='test',
            split_train='train',
            split_validation='validation',
            output_dir=f'model_ckpt/{model_alias}'
        )
        trainer.train(
            eval_step=500,
            n_trials=10,
            ray_result_dir=f"ray_results/{model_alias}",
            search_range_lr=[1e-6, 1e-4],
            search_range_epoch=[1, 6],
            search_list_batch=[8, 16, 32],
            down_sample_size_train=5000,
            down_sample_size_validation=2000
        )
        trainer.save_model()
        trainer.evaluate()
        trainer.push_to_hub(
            hf_organization='cardiffnlp',
            model_alias=f'{model_alias}'
        )

        # sample prediction
        output = trainer.predict(sample)
        pprint(f"Sample Prediction: {language_model} ({task})")
        for s, p in zip(sample, output):
            pprint(s)
            pprint(p)

        # clean up logs
        shutil.rmtree(f'model_ckpt/{model_alias}')
        shutil.rmtree(f"ray_results/{model_alias}")
        shutil.rmtree(model_alias)


# Summarize result
tmp_dir = 'metric_files'
os.makedirs(tmp_dir, exist_ok=True)


def download(filename: str, url: str):
    try:
        with open(filename) as f:
            json.load(f)
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            r = requests.get(url)
            f.write(r.content)
    try:
        with open(filename) as f:
            return json.load(f)
    except Exception:
        os.remove(filename)
        return None

summary = []
models = []
for language_model in lms:
    for task in ["irony", "offensive", "emoji", "emotion", "sentiment", "hate"]:
        model_alias = f"{os.path.basename(language_model)}-{task}"
        metric = download(
            f"{tmp_dir}/{model_alias}.json",
            f"https://huggingface.co/cardiffnlp/{model_alias}/raw/main/metric.json"
        )
        if metric is None:
            continue
        metric.update({
            "link": f"[cardiffnlp/{model_alias}](https://huggingface.co/cardiffnlp/{model_alias})",
            "language_model": f"[{language_model}](https://huggingface.co/{language_model})",
            "task": task
        })
        summary.append(metric)
df = pd.DataFrame(summary)[['task', 'language_model', 'eval_f1', 'eval_f1_macro', 'eval_accuracy', 'link']]
df = df.sort_values(by=['task', 'language_model']).round(2)
print(df.to_markdown(index=False))
df.to_csv('test_trainer_tweet_eval.csv', index=False)
