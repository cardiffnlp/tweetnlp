""" UnitTest """
import unittest
import logging

import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

SAMPLE_NER = ['Jacob Collier is a Grammy-awarded English artist from London.'] * 3
SAMPLE_CLASSIFICATION = {
    'sentiment': ["How many more days until opening day? 😩"] * 3,
    'offensive': ["All two of them taste like ass."] * 3,
    'irony': ["If you wanna look like a badass, have drama on social media"] * 3,
    'hate': ["Whoever just unfollowed me you a bitch"] * 3,
    'emotion': ["I love swimming for the same reason I love meditating...the feeling of weightlessness."] * 3,
    'emoji': ["Beautiful sunset last night from the pontoon @ Tupper Lake, New York"] * 3,
    "topic_classification": [
        'Jacob Collier is a Grammy-awarded English artist from London.',
        "I love swimming for the same reason I love meditating...the feeling of weightlessness.",
        "Beautiful sunset last night from the pontoon @ Tupper Lake, New York"
    ]
}
LM = 'distilbert-base-uncased'


class Test(unittest.TestCase):
    """ Test """

    # def test_model_ner(self):
    #     model = tweetnlp.load_model('ner')
    #     outs = model.predict(SAMPLE_NER)
    #     assert len(outs) == len(SAMPLE_NER), f"{len(outs)} != {len(SAMPLE_NER)}"
    #     for out in outs:
    #         assert all(len(list(i.keys())) == 2 and 'type' in i.keys() for i in out), out
    #         assert all(len(list(i.keys())) == 2 and 'entity' in i.keys() for i in out), out
    #
    #     outs = model.predict(SAMPLE_NER, batch_size=2)
    #     assert len(outs) == len(SAMPLE_NER), f"{len(outs)} != {len(SAMPLE_NER)}"
    #     for out in outs:
    #         assert all(len(list(i.keys())) == 2 and 'type' in i.keys() for i in out), out
    #         assert all(len(list(i.keys())) == 2 and 'entity' in i.keys() for i in out), out
    #
    #     outs = model.predict(SAMPLE_NER, return_probability=True)
    #     assert len(outs) == len(SAMPLE_NER), f"{len(outs)} != {len(SAMPLE_NER)}"
    #     for out in outs:
    #         assert all(len(list(i.keys())) == 3 and 'type' in i.keys() for i in out), out
    #         assert all(len(list(i.keys())) == 3 and 'entity' in i.keys() for i in out), out
    #         assert all(len(list(i.keys())) == 3 and 'probability' in i.keys() for i in out), out
    #
    #     out = model.predict(SAMPLE_NER[0])
    #     assert all(len(list(i.keys())) == 2 and 'type' in i.keys() for i in out), out
    #     assert all(len(list(i.keys())) == 2 and 'entity' in i.keys() for i in out), out
    #
    #     out = model.predict(SAMPLE_NER[0], return_probability=True)
    #     assert all(len(list(i.keys())) == 3 and 'type' in i.keys() for i in out), out
    #     assert all(len(list(i.keys())) == 3 and 'entity' in i.keys() for i in out), out
    #     assert all(len(list(i.keys())) == 3 and 'probability' in i.keys() for i in out), out

    def test_model_classification(self):
        for task, sample in SAMPLE_CLASSIFICATION.items():
            dataset, label_to_id = tweetnlp.load_dataset(task, multilingual=True, task_language="all")
            trainer_instance = tweetnlp.load_trainer(task)
            trainer = trainer_instance(
                language_model=LM,
                dataset=dataset,
                label_to_id=label_to_id,
                max_length=128,
                split_test='test',
                split_train='train',
                split_validation='validation',
                output_dir='tmp'
            )
            out = trainer.predict(SAMPLE_CLASSIFICATION[task][0])
            print(out)


if __name__ == "__main__":
    unittest.main()
