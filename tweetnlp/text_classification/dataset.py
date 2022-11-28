from datasets import load_dataset
from ..util import get_label2id

DEFAULT_DATASETS_TEXT_CLASSIFICATION = {
    "offensive": {'default': ["tweet_eval", "offensive"]},
    "irony": {'default': ["tweet_eval", "irony"]},
    "hate": {'default': ["tweet_eval", "hate"]},
    'emotion': {'default': ["tweet_eval", "emotion"]},
    'emoji': {'default': ["tweet_eval", "emoji"]},
    'stance_abortion': {'default': ["tweet_eval", "stance_abortion"]},
    'stance_atheism': {'default': ["tweet_eval", "stance_atheism"]},
    'stance_climate': {'default': ["tweet_eval", "stance_climate"]},
    'stance_feminist': {'default': ["tweet_eval", "stance_feminist"]},
    'stance_hillary': {'default': ["tweet_eval", "stance_hillary"]},
    "sentiment": {
        'default': ["tweet_eval", "sentiment"],
        "multilingual": ["cardiffnlp/tweet_sentiment_multilingual", None]
    },
    "topic_classification": {
        'default': ["cardiffnlp/tweet_topic_single", None],
        "multi_label": ["cardiffnlp/tweet_topic_multi", None]
    }
}


def load_dataset_text_classification(
        task_type: str = None,
        task_language: str = 'english',
        multi_label: bool = False,
        multilingual: bool = False,
        dataset_type: str = None,
        dataset_name: str = None,
        use_auth_token: bool = False):
    if task_type is not None:
        assert task_type in DEFAULT_DATASETS_TEXT_CLASSIFICATION, f"unknown task {task_type}. task type should be in " \
                                                                  f"{DEFAULT_DATASETS_TEXT_CLASSIFICATION.keys()}"
        if multilingual:
            task_name = 'multilingual'
        elif multi_label:
            task_name = 'multi_label'
        else:
            task_name = 'default'
        assert task_name in DEFAULT_DATASETS_TEXT_CLASSIFICATION[task_type], \
            f'unknown task name {task_name}. available task names are {DEFAULT_DATASETS_TEXT_CLASSIFICATION[task_type].keys()}'
        dataset_type, dataset_name = DEFAULT_DATASETS_TEXT_CLASSIFICATION[task_type][task_name]
        if multilingual:
            dataset_name = task_language
    else:
        assert dataset_type, "either of task_type or dataset_type should be specified"
    dataset = load_dataset(dataset_type, dataset_name, use_auth_token=use_auth_token)
    dataset.dataset_type = dataset_type
    dataset.dataset_name = dataset_name
    return dataset, get_label2id(dataset)
