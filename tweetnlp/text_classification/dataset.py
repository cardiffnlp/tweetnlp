from datasets import load_dataset

config = {
    "stance": {
        "label": {"default": ["none", "against", "favor"]},
        "dataset_type": {'default': "tweet_eval"},
        "dataset_name": {'default': "stance"}
    },
    "sentiment": {
        "label": {
            "default": ["negative", "neutral", "positive"],
            "multilingual": ["negative", "neutral", "positive"]},
        "dataset_type": {
            'default': "tweet_eval",
            "multilingual": "TODO https://github.com/cardiffnlp/xlm-t/tree/main/data/sentiment",
        },
        "dataset_name": {
            'default': "sentiment",
            "multilingual": "TODO https://github.com/cardiffnlp/xlm-t/tree/main/data/sentiment",
        }
    },
    "offensive": {
        "label": {"default": ["non-offensive", "offensive"]},
        "dataset_type": {'default': "tweet_eval"},
        "dataset_name": {'default': "offensive"}
    },
    "irony": {
        "label": {"default": ["non_irony", "irony"]},
        "dataset_type": {'default': "tweet_eval"},
        "dataset_name": {'default': "irony"}
    },
    "hate": {
        "label": {"default": ["non-hate", "hate"]},
        "dataset_type": {'default': "tweet_eval"},
        "dataset_name": {'default': "hate"}
    },
    'emoji': {
        "label": {
            "default": [
                "❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘", "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁",
                "🎄", "📸", "😜"
            ]
        },
        "dataset_type": {'default': "tweet_eval"},
        "dataset_name": {'default': "emoji"}
    },
    'emotion': {
        "label": {'default': ["anger", "joy", "optimism", "sadness"]},
        "dataset_type": {'default': "tweet_eval"},
        "dataset_name": {'default': "emotion"}
    },
    "topic_classification": {
        "label": {
            "default": ["arts_&_culture", "business_&_entrepreneurs", "pop_culture", "daily_life", "sports_&_gaming", "science_&_technology"],
            "multi_label": [
                "arts_&_culture", "business_&_entrepreneurs", "celebrity_&_pop_culture", "diaries_&_daily_life", "family",
                "fashion_&_style", "film_tv_&_video", "fitness_&_health", "food_&_dining", "gaming",
                "learning_&_educational", "music", "news_&_social_concern", "other_hobbies", "relationships",
                "science_&_technology", "sports", "travel_&_adventure", "youth_&_student_life"
            ]
        },
        "dataset_type": {
            "default": "cardiffnlp/tweet_topic_single",
            "multi_label": "cardiffnlp/tweet_topic_multi"
        },
        "dataset_name": {
            "default": None,
            "multi_label": None
        },
    }
}


def get_dataset(task_type: str = None,
                multi_label: bool = False,
                multilingual: bool = False,
                dataset_type: str = None,
                dataset_name: str = None):
    if task_type is not None:
        assert task_type in config, f"unknown task {task_type}. task type should be in {config.keys()}"
        if multilingual:
            task_name = 'multilingual'
        elif multi_label:
            task_name = 'multi_label'
        else:
            task_name = 'default'
        assert task_name in config[task_type], \
            f'unknown task name {task_name}. available task names are {config[task_type].keys()}'
        label2id = {k: n for n, k in enumerate(config[task_type][task_name]['label'])}
        dataset_type = config[task_type][task_name]['dataset_type']
        dataset_name = config[task_type][task_name]['dataset_name']
    else:
        assert dataset_type, "either of task_type or dataset_type should be specified"
    if dataset_name is None:
        dataset = load_dataset(dataset_type, dataset_name)
    else:
        dataset = load_dataset(dataset_type)