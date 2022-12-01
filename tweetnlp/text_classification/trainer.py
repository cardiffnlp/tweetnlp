import json
import logging
import os
import math
import shutil
import multiprocessing
from os.path import join as pj
from typing import Dict, List

import torch
import numpy as np
from huggingface_hub import create_repo
from datasets import load_metric
from datasets.dataset_dict import DatasetDict
from transformers import TrainingArguments, Trainer
from ray import tune

from .model import Classifier
from .readme_template import get_readme
from ..util import load_model


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainerTextClassification:

    def __init__(self,
                 language_model: str,
                 dataset: DatasetDict,
                 label_to_id: Dict,
                 max_length: int = 128,
                 split_train: str = 'train',
                 split_test: str = None,
                 split_validation: str = None,
                 use_auth_token: bool = False,
                 multi_label: bool = False,
                 dataset_name: str = None,
                 dataset_type: str = None,
                 output_dir: str = None):
        logging.info(f'TrainerTextClassification: {language_model}, {dataset}')
        self.dataset = dataset
        if hasattr(self.dataset, 'dataset_name'):
            self.dataset_name = self.dataset.dataset_name
        else:
            self.dataset_name = dataset_name
        if hasattr(self.dataset, 'dataset_type'):
            self.dataset_type = self.dataset.dataset_type
        else:
            self.dataset_type = dataset_type

        self.language_model = language_model
        self.use_auth_token = use_auth_token
        self.multi_label = multi_label
        self.output_dir = output_dir
        self.model_config = {
            'label2id': label_to_id,
            'id2label': {v: k for k, v in label_to_id.items()},
            "num_labels": len(label_to_id)
        }
        if self.multi_label:
            self.model_config['problem_type'] = "multi_label_classification"
        self.config, self.tokenizer, self.model = load_model(
            self.language_model,
            task='sequence_classification',
            use_auth_token=self.use_auth_token,
            config_argument=self.model_config
        )
        self.max_length = max_length
        self.tokenized_datasets = self.dataset.map(
            lambda x: self.tokenizer(x["text"], padding="max_length", truncation=True, max_length=max_length),
            batched=True)
        # setup metrics
        self.compute_metric_search, self.compute_metric_all = self.get_metrics()
        self.best_model_path = pj(self.output_dir, 'best_model')
        self.best_run_hyperparameters_path = pj(self.output_dir, 'best_run_hyperparameters.json')

        self.split_test = split_test
        self.split_train = split_train
        self.split_validation = split_validation
        self.trainer = None
        self.classifier = None

    @property
    def export_file(self):
        assert self.output_dir is not None, "output_dir is not defined"
        return f"{self.output_dir}/metric.json"

    def get_metrics(self):
        if self.multi_label:
            metric_accuracy = load_metric("accuracy", "multilabel")
            metric_f1 = load_metric("f1", "multilabel")
        else:
            metric_accuracy = load_metric("accuracy")
            metric_f1 = load_metric("f1")

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def compute_metric_search(eval_pred):
            logits, labels = eval_pred
            if self.multi_label:
                predictions = np.array([[int(sigmoid(j) > 0.5) for j in i] for i in logits])
            else:
                predictions = np.argmax(logits, axis=-1)
            return metric_f1.compute(predictions=predictions, references=labels, average='micro')

        def compute_metric_all(eval_pred):
            logits, labels = eval_pred
            if self.multi_label:
                predictions = np.array([[int(sigmoid(j) > 0.5) for j in i] for i in logits])
            else:
                predictions = np.argmax(logits, axis=-1)
            return {
                'f1': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
                'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
                'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
            }

        return compute_metric_search, compute_metric_all

    def train(self,
              output_dir: str = None,
              random_seed: int = 42,
              eval_step: int = 100,
              n_trials: int = 10,
              split_train: str = None,
              split_validation: str = None,
              parallel_cpu: bool = False,
              search_range_lr: List = None,
              search_range_epoch: List = None,
              search_list_batch: List = None,
              ray_result_dir: str = 'ray_results',
              down_sample_size_train: int = None,
              down_sample_size_validation: int = None,
              training_arguments: TrainingArguments = None):
        if output_dir is not None:
            self.output_dir = output_dir
        assert self.output_dir is not None, "output_dir should be specified."
        os.makedirs(self.output_dir, exist_ok=True)
        if split_train is not None:
            self.split_train = split_train
        if split_validation is not None:
            self.split_validation = split_validation
        assert self.split_train in self.dataset.keys(),\
            f"train split not found: {self.split_train} is not in {self.dataset.keys()}"
        # setup trainer
        full_train_dataset = self.tokenized_datasets[self.split_train]
        search_train_dataset = full_train_dataset
        if down_sample_size_train is not None and down_sample_size_train < len(
                self.tokenized_datasets[self.split_train]):
            tmp = self.tokenized_datasets[self.split_train]
            tmp = tmp.shuffle(random_seed)
            search_train_dataset = tmp.select(list(range(down_sample_size_train)))
        if self.split_validation is None:
            logging.warning('setup trainer without hyperparameter tuning. (provide `split_validation` for hyperparameter search)')
            training_arguments = TrainingArguments(
                    output_dir=self.output_dir,
                    evaluation_strategy="no",
                    eval_steps=eval_step,
                    seed=random_seed
                ) if training_arguments is None else training_arguments
            self.trainer = Trainer(
                model=self.model,
                args=training_arguments,
                train_dataset=full_train_dataset
            )
        else:
            assert self.split_validation in self.dataset.keys(), \
                f"validation split not found: {self.split_validation} is not in {self.dataset.keys()}"
            if down_sample_size_validation is not None and down_sample_size_validation < len(
                    self.tokenized_datasets[self.split_validation]):
                tmp = self.tokenized_datasets[self.split_validation]
                tmp = tmp.shuffle(random_seed)
                self.tokenized_datasets[self.split_validation] = tmp.select(list(range(down_sample_size_validation)))

            self.trainer = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir=self.output_dir,
                    evaluation_strategy="steps",
                    eval_steps=eval_step,
                    seed=random_seed
                ),
                train_dataset=search_train_dataset,
                eval_dataset=self.tokenized_datasets[self.split_validation],
                compute_metrics=self.compute_metric_search,
                model_init=lambda x: load_model(
                    self.language_model,
                    return_dict=True,
                    task='sequence_classification',
                    use_auth_token=self.use_auth_token,
                    model_argument=self.model_config,
                )
            )
            # define search space
            logging.info('define search space')
            search_range_lr = [1e-6, 1e-4] if search_range_lr is None else search_range_lr
            assert len(search_range_lr) == 2, f"len(search_range_lr) should be 2: {search_range_lr}"
            search_range_epoch = [1, 6] if search_range_epoch is None else search_range_epoch
            assert len(search_range_epoch) == 2, f"len(search_range_epoch) should be 2: {search_range_epoch}"
            search_list_batch = [4, 8, 16, 32, 64] if search_list_batch is None else search_list_batch
            search_space = {
                "learning_rate": tune.loguniform(search_range_lr[0], search_range_lr[1]),
                "num_train_epochs": tune.choice(list(range(search_range_epoch[0], search_range_epoch[1]))),
                "per_device_train_batch_size": tune.choice(search_list_batch)
            }
            resources_per_trial = {'cpu': multiprocessing.cpu_count() if parallel_cpu else 1, "gpu": torch.cuda.device_count()}
            logging.info(f'run on `{resources_per_trial["cpu"]}` cpus and `{resources_per_trial["gpu"]}` gpus')
            # run parameter search
            logging.info("start parameter search")
            best_run = self.trainer.hyperparameter_search(
                hp_space=lambda x: search_space,
                local_dir=ray_result_dir,
                direction="maximize",
                backend="ray",
                n_trials=n_trials,
                resources_per_trial=resources_per_trial
            )
            # finetuning with the best config
            with open(self.best_run_hyperparameters_path, 'w') as f:
                json.dump(best_run.hyperparameters, f)
            logging.info(f"fine-tuning with the best config: {best_run} (saved at {self.best_run_hyperparameters_path})")
            for n, v in best_run.hyperparameters.items():
                setattr(self.trainer.args, n, v)
            setattr(self.trainer, "train_dataset", full_train_dataset)
            setattr(self.trainer.args, "evaluation_strategy", 'no')
        self.trainer.train()
        logging.info('training finished')
        self.model = self.trainer.model

    def predict(self,
                text: str or List,
                batch_size: int = None,
                return_probability: bool = False,
                skip_preprocess: bool = False):
        if self.trainer is None:
            logging.warning("model is not trained.")
        if self.classifier is None:
            self.classifier = Classifier(loaded_model_config_tokenizer={
                "model": self.model, "tokenizer": self.tokenizer, "config": self.config
            })
        return self.classifier.predict(
            text, batch_size=batch_size, return_probability=return_probability, skip_preprocess=skip_preprocess)

    def save_model(self, model_path: str = None):
        assert self.trainer is not None, "model is not trained."
        if model_path is not None:
            self.best_model_path = model_path
            os.makedirs(model_path, exist_ok=True)
        self.trainer.save_model(self.best_model_path)
        self.tokenizer.save_pretrained(self.best_model_path)
        logging.info(f"best model saved at {self.best_model_path}")

    def evaluate(self, split_test: str = None, output_dir: str = None):
        if self.trainer is None:
            logging.warning("model is not trained.")
        if output_dir is not None:
            self.output_dir = output_dir
        assert self.output_dir is not None, "output_dir should be specified."
        if split_test is not None:
            self.split_test = split_test
        assert self.split_test is not None and self.split_test in self.dataset.keys(), \
            f"test split not found: {self.split_test} is not in {self.dataset.keys()}"
        logging.info('model evaluation')
        # self.model = load_model(
        #     model=self.language_model if not os.path.exists(self.best_model_path) else self.best_model_path,
        #     model_only=True,
        #     task='sequence_classification',
        #     use_auth_token=self.use_auth_token,
        #     model_argument=self.model_config
        # )
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=self.output_dir, evaluation_strategy="no"),
            eval_dataset=self.tokenized_datasets[self.split_test],
            compute_metrics=self.compute_metric_all
        )
        result = {k: v for k, v in trainer.evaluate().items()}
        logging.info(json.dumps(result, indent=4))
        with open(self.export_file, 'w') as f:
            json.dump(result, f)
        return result

    def push_to_hub(self,
                    hf_organization: str,
                    model_alias: str,
                    split_train: str = None,
                    split_validation: str = None,
                    split_test: str = None,
                    dataset_name: str = None,
                    dataset_type: str = None,
                    output_dir: str = None):
        if self.trainer is None:
            logging.warning("model is not trained.")
        if dataset_name is not None:
            self.dataset_name = dataset_name
        if dataset_type is not None:
            self.dataset_type = dataset_type
        if output_dir is not None:
            self.output_dir = output_dir
        assert self.output_dir is not None, "output_dir should be specified."
        if split_train is not None:
            self.split_train = split_train
        if split_validation is not None:
            self.split_validation = split_validation
        if split_test is not None:
            self.split_test = split_test
        logging.info('uploading to huggingface')
        url = create_repo(model_alias, organization=hf_organization, exist_ok=True)
        args = {"use_auth_token": self.use_auth_token, "repo_url": url, "organization": hf_organization}
        self.model.push_to_hub(model_alias, **args)
        self.tokenizer.push_to_hub(model_alias, **args)
        readme = get_readme(
            model_name=f"{hf_organization}/{model_alias}",
            metric_file=self.export_file,
            dataset_name=self.dataset_name,
            dataset_type=self.dataset_type,
            language_model=self.language_model,
            split_test=self.split_test,
            split_validation=self.split_validation,
            split_train=self.split_train,
        )
        with open(f"{model_alias}/README.md", "w") as f:
            f.write(readme)
        if os.path.exists(self.best_run_hyperparameters_path):
            shutil.copy2(self.best_run_hyperparameters_path, pj(model_alias, 'best_run_hyperparameters.json'))
        os.system(
            f"cd {model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
