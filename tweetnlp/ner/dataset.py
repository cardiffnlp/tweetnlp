from datasets import load_dataset
from ..util import get_label2id

# DEFAULT_DATASET_TYPES_NER = {
#     "tweetner7": {'default': ["tner/tweetner7", None]},
#     "tweebank_ner": {'default': ["tner/tweebank_ner", None]},
#     "btc": {'default': ["tner/btc", None]},
#     "ttc": {'default': ["tner/ttc", None]}
# }

DEFAULT_DATASET_TYPE_NER = "tner/tweetner7"


def load_dataset_ner(
        task_type: str = 'ner',
        dataset_type: str = None,
        dataset_name: str = None,
        use_auth_token: bool = False):
    assert task_type == 'ner', task_type
    dataset_type = DEFAULT_DATASET_TYPE_NER if dataset_type is None else dataset_type
    dataset = load_dataset(dataset_type, dataset_name, use_auth_token=use_auth_token)
    dataset.dataset_type = dataset_type
    dataset.dataset_name = dataset_name
    return dataset, get_label2id(dataset, label_name='tags')
