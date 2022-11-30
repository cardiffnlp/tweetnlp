import os.path
import logging
import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

##############################
# basic classification tasks #
##############################
for language_model in ['cardiffnlp/twitter-roberta-base-dec2021', 'cardiffnlp/twitter-roberta-base-2021-124m']:
    for task in ["irony", "offensive", "emoji", "emotion", "sentiment", "hate"]:
        dataset, label_to_id = tweetnlp.load_dataset(task)
        trainer_class = tweetnlp.load_trainer(task)
        trainer = trainer_class(
            language_model=language_model,
            dataset=dataset,
            label_to_id=label_to_id,
            max_length=128,
            split_test='test',
            split_train='train',
            split_validation='validation',
            output_dir=f'model_ckpt/{os.path.basename(language_model)}_{task}'
        )
        trainer.train(
            eval_step=50,
            n_trials=10,
            ray_result_dir=,
            search_range_lr=[1e-6, 1e-4],
            search_range_epoch=[1, 6],
            search_list_batch=[4, 8, 16, 32, 64],
            down_sample_size_train=5000,
            down_sample_size_validation=2000
        )
        trainer.save_model()
        trainer.evaluate()
        trainer.push_to_hub(hf_organization='cardiffnlp', model_alias=f'{os.path.basename(language_model)}-{task}')

