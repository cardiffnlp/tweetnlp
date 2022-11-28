""" UnitTest """
import unittest
import logging
from pprint import pprint
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
        model = tweetnlp.load_model('ner')
        preds = model.ner(SAMPLE)
        pprint(preds)
        preds = model.predict(SAMPLE, return_probability=True, return_position=True)
        pprint(preds)


if __name__ == "__main__":
    unittest.main()
