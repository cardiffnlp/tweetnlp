import os
import logging
import shutil
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


##########################
# multilingual sentiment #
##########################
lms = ["cardiffnlp/twitter-xlm-roberta-base", 'bert-base-multilingual-cased', "xlm-roberta-base"]
for language_model in lms:
    model_alias = f"{os.path.basename(language_model)}-sentiment-multilingual"
    dataset, label_to_id = tweetnlp.load_dataset("sentiment", multilingual=True, task_language="all")
    trainer_class = tweetnlp.load_trainer("sentiment")

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
        eval_step=1000,
        n_trials=10,
        ray_result_dir=f"ray_result/{model_alias}",
        # parallel_cpu=True,
        search_range_lr=[1e-6, 1e-4],
        search_range_epoch=[1, 6],
        search_list_batch=[4, 8, 16, 32],
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
    pprint(f"Sample Prediction: {language_model}")
    for s, p in zip(sample, output):
        pprint(s)
        pprint(p)

    # clean up logs
    shutil.rmtree(f'model_ckpt/{model_alias}')
    shutil.rmtree(f"ray_result/{model_alias}")
    shutil.rmtree(model_alias)

########################
# topic_classification #
########################
lms = ['cardiffnlp/twitter-roberta-base-dec2021', 'cardiffnlp/twitter-roberta-base-2021-124m', 'roberta-base']
for language_model in lms:
    for multi_label in [True, False]:
        model_alias = f"{os.path.basename(language_model)}-topic-{'multi' if multi_label else 'single'}"
        dataset, label_to_id = tweetnlp.load_dataset("topic_classification", multi_label=multi_label)
        trainer_class = tweetnlp.load_trainer("topic_classification")

        # setup trainer
        trainer = trainer_class(
            language_model=language_model,
            dataset=dataset,
            label_to_id=label_to_id,
            max_length=128,
            split_test='test_2021',
            split_train='train_all',
            split_validation='validation_2021',
            output_dir=f'model_ckpt/{model_alias}'
        )
        trainer.train(
            eval_step=1000,
            n_trials=10,
            ray_result_dir=f"ray_result/{model_alias}",
            parallel_cpu=True,
            search_range_lr=[1e-6, 1e-4],
            search_range_epoch=[1, 6],
            search_list_batch=[4, 8, 16, 32],
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
        pprint(f"Sample Prediction: {language_model}")
        for s, p in zip(sample, output):
            pprint(s)
            pprint(p)

        # clean up logs
        shutil.rmtree(f'model_ckpt/{model_alias}')
        shutil.rmtree(f"ray_result/{model_alias}")
        shutil.rmtree(model_alias)
