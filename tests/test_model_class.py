""" UnitTest """
import unittest
import logging

import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

SAMPLE = {
    # 'ner': ['Jacob Collier is a Grammy-awarded English artist from London.'],
    # 'sentiment': ["How many more days until opening day? 😩"],
    # 'offensive': ["All two of them taste like ass."],
    # 'irony': ["If you wanna look like a badass, have drama on social media"],
    # 'hate': ["Whoever just unfollowed me you a bitch"],
    # 'emotion': ["I love swimming for the same reason I love meditating...the feeling of weightlessness."],
    'emoji': ["Beautiful sunset last night from the pontoon @ Tupper Lake, New York"],
    # 'language_model': [
    #     "So glad I'm <mask> vaccinated.",
    #     "I keep forgetting to bring a <mask>.",
    #     "Looking forward to watching <mask> Game tonight!"
    # ]
}


class Test(unittest.TestCase):
    """ Test """

    def test_model(self):
        for task, sample in SAMPLE.items():
            print(task)
            model = tweetnlp.load(task)
            print(sample)
            o = model.predict(sample)
            print(o)


if __name__ == "__main__":
    unittest.main()
