import shutil
import logging
import os
import tweetnlp
from pprint import pprint

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

##########################
# multilingual sentiment #
##########################
task = "sentiment"
language_model = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# language_model = 'bert-base-multilingual-cased'
# language_model = "xlm-roberta-base"
model_alias = f'{os.path.basename(language_model)}-{task}-multilingual'
dataset, label_to_id = tweetnlp.load_dataset(task, multilingual=True, task_language="all")
trainer_class = tweetnlp.load_trainer(task)
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
    eval_step=1000,
    n_trials=10,
    search_range_lr=[1e-6, 1e-4],
    search_range_epoch=[1, 6],
    search_list_batch=[4, 8, 16, 32, 64],
    down_sample_size_train=5000,
    down_sample_size_validation=2000
)
trainer.save_model()
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=model_alias)
pred = trainer.predict(["天気が良いとやっぱり気持ち良いなあ✨", "Yes, including Medicare and social security saving👍"])
pprint(pred)
pred = trainer.predict(["天気が良いとやっぱり気持ち良いなあ✨", "Yes, including Medicare and social security saving👍"],
                       return_probability=True)
pprint(pred)

#######################################
# topic_classification (single label) #
#######################################
task = "topic_classification"
dataset, label_to_id = tweetnlp.load_dataset(task)
trainer_class = tweetnlp.load_trainer(task)
trainer = trainer_class(
    language_model='distilbert-base-uncased',
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_test='test_2021',
    split_train='train_all',
    split_validation='validation_2021',
    output_dir=f'model_ckpt/{task}'
)
trainer.train(
    eval_step=50,
    n_trials=10,
    search_range_lr=[1e-6, 1e-4],
    search_range_epoch=[1, 6],
    search_list_batch=[4, 8, 16, 32, 64]
)
trainer.save_model()
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'distilbert-{task}-single-all')
shutil.rmtree("ray_result")
shutil.rmtree("model_ckpt")

######################################
# topic_classification (multi label) #
######################################
task = "topic_classification"
dataset, label_to_id = tweetnlp.load_dataset(task, multilabel=True)
trainer_class = tweetnlp.load_trainer(task)
trainer = trainer_class(
    language_model='distilbert-base-uncased',
    multi_label=True,
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_test='test_2021',
    split_train='train_all',
    split_validation='validation_2021',
    output_dir=f'model_ckpt/{task}-multi-label'
)
trainer.train(
    eval_step=50,
    n_trials=10,
    search_range_lr=[1e-6, 1e-4],
    search_range_epoch=[1, 6],
    search_list_batch=[4, 8, 16, 32, 64]
)
trainer.save_model()
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'distilbert-{task}-multi-all')
shutil.rmtree("ray_result")
shutil.rmtree("model_ckpt")
