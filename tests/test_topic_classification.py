""" UnitTest """
import unittest
import logging

import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

SAMPLE = [
    'Jacob Collier is a Grammy-awarded English artist from London.',
    "How many more days until opening day? 😩",
    "All two of them taste like ass.",
    "If you wanna look like a badass, have drama on social media",
    "Whoever just unfollowed me you a bitch",
    "I love swimming for the same reason I love meditating...the feeling of weightlessness.",
    "Beautiful sunset last night from the pontoon @ Tupper Lake, New York"
]


class Test(unittest.TestCase):
    """ Test """

    def test_model(self):
        model = tweetnlp.load('topic_classification')
        preds = model.topic(SAMPLE)
        for text, pred in zip(SAMPLE, preds):
            print('MULTI-CLASS', text, pred)

    def test_model_single_label(self):
        model = tweetnlp.load('topic_classification', single_class=True)
        preds = model.topic(SAMPLE)
        for text, pred in zip(SAMPLE, preds):
            print('SINGLE-CLASS', text, pred)


if __name__ == "__main__":
    unittest.main()
