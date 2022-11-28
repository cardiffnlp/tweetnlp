from datasets import load_dataset
from ..util import get_label2id

DEFAULT_DATASETS_NER = {
    "ner": {'default': ["tner/tweetner7", None]},
    "tweetner7": {'default': ["tner/tweetner7", None]},
    "tweebank_ner": {'default': ["tner/tweebank_ner", None]},
    "btc": {'default': ["tner/btc", None]},
    "ttc": {'default': ["tner/ttc", None]}
}


def load_dataset_ner(
        task_type: str = None,
        dataset_type: str = None,
        dataset_name: str = None,
        use_auth_token: bool = False):
    if task_type is not None:
        assert task_type in DEFAULT_DATASETS_NER, f"unknown task {task_type}. task type should be in " \
                                                                  f"{DEFAULT_DATASETS_NER.keys()}"
        task_name = 'default'
        dataset_type, dataset_name = DEFAULT_DATASETS_NER[task_type][task_name]
    else:
        assert dataset_type, "either of task_type or dataset_type should be specified"
    dataset = load_dataset(dataset_type, dataset_name, use_auth_token=use_auth_token)
    return dataset, get_label2id(dataset, label_name='tags')
