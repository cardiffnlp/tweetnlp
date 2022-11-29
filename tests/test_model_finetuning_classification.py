import shutil
import logging
from glob import glob
import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

##############################
# basic classification tasks #
##############################
# for task in ["irony", "offensive", "emoji", "emotion", "sentiment", "hate"]:
for task in ["offensive", "emoji", "emotion", "sentiment", "hate"]:
    dataset, label_to_id = tweetnlp.load_dataset(task)
    trainer_class = tweetnlp.load_trainer(task)
    trainer = trainer_class(
        language_model='cardiffnlp/twitter-roberta-base-dec2021',
        dataset=dataset,
        label_to_id=label_to_id,
        max_length=128,
        split_test='test',
        split_train='train',
        split_validation='validation',
        output_dir=f'model_ckpt/{task}'
    )
    trainer.train(
        eval_step=50,
        n_trials=10,
        search_range_lr=[1e-6, 1e-4],
        search_range_epoch=[1, 6],
        search_list_batch=[4, 8, 16, 32, 64]
    )
    trainer.evaluate()
    trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'twitter-roberta-base-dec2021-{task}')
    shutil.rmtree("ray_result")
    [shutil.rmtree(i) for i in glob(f'model_ckpt/{task}/checkpoint-*')]
    shutil.rmtree(f'model_ckpt/{task}/runs')

##########################
# multilingual sentiment #
##########################
task = "sentiment"
dataset, label_to_id = tweetnlp.load_dataset(task, multilingual=True)
trainer_class = tweetnlp.load_trainer(task)
trainer = trainer_class(
    language_model='bert-base-multilingual-cased',
    multi_label=True,
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_test='test_2021',
    split_train='train_all',
    split_validation='validation_2021',
    output_dir=f'model_ckpt/{task}-multilingual'
)
trainer.train(
    eval_step=50,
    n_trials=10,
    search_range_lr=[1e-6, 1e-4],
    search_range_epoch=[1, 6],
    search_list_batch=[4, 8, 16, 32, 64]
)
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'mbert-{task}-multilingual')
shutil.rmtree("ray_result")
[shutil.rmtree(i) for i in glob(f'model_ckpt/{task}-multilingual/checkpoint-*')]
shutil.rmtree(f'model_ckpt/{task}-multilingual/runs')

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
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'distilbert-{task}-single-all')
shutil.rmtree("ray_result")
[shutil.rmtree(i) for i in glob(f'model_ckpt/{task}/checkpoint-*')]
shutil.rmtree(f'model_ckpt/{task}/runs')

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
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'distilbert-{task}-multi-all')
shutil.rmtree("ray_result")
[shutil.rmtree(i) for i in glob(f'model_ckpt/{task}-multi-label/checkpoint-*')]
shutil.rmtree(f'model_ckpt/{task}-multi-label/runs')

