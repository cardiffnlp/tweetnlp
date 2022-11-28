# import argparse
# import json
# import logging
# import os
# import math
# import shutil
# import urllib.request
# import multiprocessing
# from os.path import join as pj
# from typing import Dict
#
# import torch
# import numpy as np
# from huggingface_hub import create_repo
# from datasets import load_dataset, load_metric
# from datasets.dataset_dict import DatasetDict
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from ray import tune
#
# from readme import get_readme
#
# from .model import load_model
#
#
# # logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
#
# PARALLEL = bool(int(os.getenv("PARALLEL", 1)))
# RAY_RESULTS = os.getenv("RAY_RESULTS", "ray_results")
# LABEL2ID = {
#     "arts_&_culture": 0,
#     "business_&_entrepreneurs": 1,
#     "celebrity_&_pop_culture": 2,
#     "diaries_&_daily_life": 3,
#     "family": 4,
#     "fashion_&_style": 5,
#     "film_tv_&_video": 6,
#     "fitness_&_health": 7,
#     "food_&_dining": 8,
#     "gaming": 9,
#     "learning_&_educational": 10,
#     "music": 11,
#     "news_&_social_concern": 12,
#     "other_hobbies": 13,
#     "relationships": 14,
#     "science_&_technology": 15,
#     "sports": 16,
#     "travel_&_adventure": 17,
#     "youth_&_student_life": 18
# }
# ID2LABEL = {v: k for k, v in LABEL2ID.items()}
#
#
# def internet_connection(host='http://google.com'):
#     try:
#         urllib.request.urlopen(host)
#         return True
#     except:
#         return False
#
#
# def get_metrics():
#     metric_accuracy = load_metric("accuracy", "multilabel")
#     metric_f1 = load_metric("f1", "multilabel")
#
#     def sigmoid(x):
#         return 1 / (1 + math.exp(-x))
#
#     # metric_f1.compute(predictions=[[0, 1, 1], [1, 1, 0]], references=[[0, 1, 1], [0, 1, 0]], average='micro')
#     # metric_accuracy.compute(predictions=[[0, 1, 1], [1, 1, 0]], references=[[0, 1, 1], [0, 1, 0]])
#
#     def compute_metric_search(eval_pred):
#         logits, labels = eval_pred
#         predictions = np.array([[int(sigmoid(j) > 0.5) for j in i] for i in logits])
#         return metric_f1.compute(predictions=predictions, references=labels, average='micro')
#
#     def compute_metric_all(eval_pred):
#         logits, labels = eval_pred
#         predictions = np.array([[int(sigmoid(j) > 0.5) for j in i] for i in logits])
#         return {
#             'f1': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
#             'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
#             'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
#         }
#     return compute_metric_search, compute_metric_all
#
#
# class TrainerTextClassification:
#
#     def __init__(self,
#                  language_model: str,
#                  dataset: DatasetDict,
#                  max_length: int,
#                  random_seed: int,
#                  eval_step: int,
#                  output_dir: str,
#                  n_trials: int):
#         self.config, self.tokenizer, self.model = load_model(language_model)
#         self.dataset = dataset
#
#         self.max_length = max_length
#         self.random_seed = random_seed
#         self.n_trials = n_trials
#         self.eval_step = eval_step
#         self.output_dir = output_dir
#
#         os.makedirs(self.output_dir, exist_ok=True)
#
#     def train(self, split_train: str = 'train', split_validation: str = 'validation'):
#
#
#
# def trainer_text_classification(
#         language_model: str,
#         dataset,
#         split_train: str,
#         split_validation: str,
#         split_test: str,
#         max_length: int,
#         random_seed: int,
#         eval_step: int,
#         output_dir: str,
#         n_trials: int,
#         push_to_hub: bool,
#         use_auth_token: bool,
#         hf_organization: str,
#         model_alias: str,
#         summary_file,
#         skip_train: bool,
#         skip_eval: bool):
#     assert summary_file.endswith('.json'), f'`--summary-file` should be a json file {summary_file}'
#     # setup data
#     dataset = load_dataset(dataset)
#     network = internet_connection()
#     # setup model
#     tokenizer = AutoTokenizer.from_pretrained(language_model, local_files_only=not network)
#     model = AutoModelForSequenceClassification.from_pretrained(
#         language_model,
#         id2label=ID2LABEL,
#         label2id=LABEL2ID,
#         num_labels=len(dataset[split_train]['label'][0]),
#         local_files_only=not network,
#         problem_type="multi_label_classification"
#     )
#     tokenized_datasets = dataset.map(
#         lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length),
#         batched=True)
#     # setup metrics
#     compute_metric_search, compute_metric_all = get_metrics()
#
#     if not skip_train:
#         # setup trainer
#         trainer = Trainer(
#             model=model,
#             args=TrainingArguments(
#                 output_dir=output_dir,
#                 evaluation_strategy="steps",
#                 eval_steps=eval_step,
#                 seed=random_seed
#             ),
#             train_dataset=tokenized_datasets[split_train],
#             eval_dataset=tokenized_datasets[split_validation],
#             compute_metrics=compute_metric_search,
#             model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
#                 language_model,
#                 return_dict=True,
#                 num_labels=len(dataset[split_train]['label'][0]),
#                 id2label=ID2LABEL,
#                 label2id=LABEL2ID
#             )
#         )
#         # parameter search
#         if PARALLEL:
#             best_run = trainer.hyperparameter_search(
#                 hp_space=lambda x: {
#                     "learning_rate": tune.loguniform(1e-6, 1e-4),
#                     "num_train_epochs": tune.choice(list(range(1, 6))),
#                     "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
#                 },
#                 local_dir=RAY_RESULTS, direction="maximize", backend="ray", n_trials=n_trials,
#                 resources_per_trial={'cpu': multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()},
#
#             )
#         else:
#             best_run = trainer.hyperparameter_search(
#                 hp_space=lambda x: {
#                     "learning_rate": tune.loguniform(1e-6, 1e-4),
#                     "num_train_epochs": tune.choice(list(range(1, 6))),
#                     "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
#                 },
#                 local_dir=RAY_RESULTS, direction="maximize", backend="ray", n_trials=n_trials
#             )
#         # finetuning
#         for n, v in best_run.hyperparameters.items():
#             setattr(trainer.args, n, v)
#         trainer.train()
#         trainer.save_model(pj(output_dir, 'best_model'))
#         best_model_path = pj(output_dir, 'best_model')
#     else:
#         best_model_path = pj(output_dir, 'best_model')
#
#     # evaluation
#     model = AutoModelForSequenceClassification.from_pretrained(
#         best_model_path,
#         num_labels=len(dataset[split_train]['label'][0]),
#         local_files_only=not network,
#         problem_type="multi_label_classification",
#         id2label=ID2LABEL,
#         label2id=LABEL2ID
#     )
#     trainer = Trainer(
#         model=model,
#         args=TrainingArguments(
#             output_dir=output_dir,
#             evaluation_strategy="no",
#             seed=random_seed
#         ),
#         train_dataset=tokenized_datasets[split_train],
#         eval_dataset=tokenized_datasets[split_test],
#         compute_metrics=compute_metric_all
#     )
#     summary_file = pj(output_dir, summary_file)
#     if not skip_eval:
#         result = {f'test/{k}': v for k, v in trainer.evaluate().items()}
#         logging.info(json.dumps(result, indent=4))
#         with open(summary_file, 'w') as f:
#             json.dump(result, f)
#
#     if push_to_hub:
#         assert hf_organization is not None, f'specify hf organization `--hf-organization`'
#         assert model_alias is not None, f'specify hf organization `--model-alias`'
#         url = create_repo(model_alias, organization=hf_organization, exist_ok=True)
#         # if not skip_train:
#         args = {"use_auth_token": use_auth_token, "repo_url": url, "organization": hf_organization}
#         trainer.model.push_to_hub(model_alias, **args)
#         tokenizer.push_to_hub(model_alias, **args)
#         if os.path.exists(summary_file):
#             shutil.copy2(summary_file, model_alias)
#         extra_desc = f"This model is fine-tuned on `{split_train}` split and validated on `{split_test}` split of tweet_topic."
#         readme = get_readme(
#             model_name=f"{hf_organization}/{model_alias}",
#             metric=summary_file,
#             language_model=model,
#             extra_desc=extra_desc
#             )
#         with open(f"{model_alias}/README.md", "w") as f:
#             f.write(readme)
#         os.system(
#             f"cd {model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
#         shutil.rmtree(f"{model_alias}")  # clean up the cloned repo
