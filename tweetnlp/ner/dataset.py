from datasets import load_dataset
from datasets.features.features import Sequence, ClassLabel

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
    if dataset_name is not None:
        dataset = load_dataset(dataset_type, dataset_name, use_auth_token=use_auth_token)
    else:
        dataset = load_dataset(dataset_type, use_auth_token=use_auth_token)

    label_info = dataset[list(dataset.keys())[0]].features['tags']
    while True:
        if type(label_info) is Sequence:
            label_info = label_info.feature
        else:
            assert type(label_info) is ClassLabel, f"Error at retrieving label information {label_info}"
            break
    label2id = {k: n for n, k in enumerate(label_info.names)}
    return dataset, label2id


if __name__ == '__main__':
    load_dataset_ner("ner")
