""" UnitTest """
import unittest
import logging

import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

SAMPLE_QUESTION = [
            'who created the post as we know it today?',
            "What is a person called is practicing heresy?"
        ]
SAMPLE_CONTEXT = [
    "'So much of The Post is Ben,' Mrs. Graham said in 1994, three years after Bradlee retired as editor. 'He created it as we know it today.'— Ed O'Keefe (@edatpost) October 21, 2014",
    "Heresy is any provocative belief or theory that is strongly at variance with established beliefs or customs. A heretic is a proponent of such claims or beliefs. Heresy is distinct from both apostasy, which is the explicit renunciation of one's religion, principles or cause, and blasphemy, which is an impious utterance or action concerning God or sacred things."
]


class Test(unittest.TestCase):
    """ Test """

    def test_model_qa(self):
        model = tweetnlp.load_model('question_answering')
        outs = model.predict(question=SAMPLE_QUESTION, context=SAMPLE_CONTEXT)
        print(outs)

    def test_model_qag(self):
        model = tweetnlp.load_model('question_answer_generation')
        outs = model.predict(SAMPLE_CONTEXT)
        print(outs)


if __name__ == "__main__":
    unittest.main()
