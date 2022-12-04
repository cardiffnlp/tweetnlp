from datasets import load_dataset


DEFAULT_DATASET_TYPE_QAG = "lmqg/qag_tweetqa"


def load_dataset_question_answer_generation(
        task_type: str = 'question_answer_generation',
        dataset_type: str = None,
        dataset_name: str = None,
        use_auth_token: bool = False):
    assert task_type == 'question_answer_generation', task_type
    dataset_type = DEFAULT_DATASET_TYPE_QAG if dataset_type is None else dataset_type
    dataset = load_dataset(dataset_type, dataset_name, use_auth_token=use_auth_token)
    dataset.dataset_type = dataset_type
    dataset.dataset_name = dataset_name
    return dataset
