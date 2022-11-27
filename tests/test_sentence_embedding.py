""" UnitTest """
import unittest
import logging

import tweetnlp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

tweet_corpus = [
    "Free, fair elections are the lifeblood of our democracy. Charges of unfairness are serious. But calling an election unfair does not make it so. Charges require specific allegations and then proof. We have neither here.",
    "Trump appointed judge Stephanos Bibas ",
    "If your members can go to Puerto Rico they can get their asses back in the classroom. @CTULocal1",
    "@PolitiBunny @CTULocal1 Political leverage, science said schools could reopen, teachers and unions protested to keep'em closed and made demands for higher wages and benefits, they're usin Covid as a crutch at the expense of life and education.",
    "Congratulations to all the exporters on achieving record exports in Dec 2020 with a growth of 18 % over the previous year. Well done &amp; keep up this trend. A major pillar of our govt's economic policy is export enhancement &amp; we will provide full support to promote export culture.",
    "@ImranKhanPTI Pakistan seems a worst country in term of exporting facilities. I am a small business man and if I have to export a t-shirt having worth of $5 to USA or Europe. Postal cost will be around $30. How can we grow as an exporting country if this situation prevails. Think about it. #PM",
    "The thing that doesn’t sit right with me about “nothing good happened in 2020” is that it ignores the largest protest movement in our history. The beautiful, powerful Black Lives Matter uprising reached every corner of the country and should be central to our look back at 2020.",
    "@JoshuaPotash I kinda said that in the 2020 look back for @washingtonpost",
    "Is this a confirmation from Q that Lin is leaking declassified intelligence to the public? I believe so. If @realDonaldTrump didn’t approve of what @LLinWood is doing he would have let us know a lonnnnnng time ago. I’ve always wondered why Lin’s Twitter handle started with “LLin” https://t.co/0G7zClOmi2",
    "@ice_qued @realDonaldTrump @LLinWood Yeah 100%",
    "Tomorrow is my last day as Senator from Alabama.  I believe our opportunities are boundless when we find common ground. As we swear in a new Congress &amp; a new President, demand from them that they do just that &amp; build a stronger, more just society.  It’s been an honor to serve you." 
    "The mask cult can’t ever admit masks don’t work because their ideology is based on feeling like a “good person”  Wearing a mask makes them a “good person” &amp; anyone who disagrees w/them isn’t  They can’t tolerate any idea that makes them feel like their self-importance is unearned",
    "@ianmSC Beyond that, they put such huge confidence in masks so early with no strong evidence that they have any meaningful benefit, they don’t want to backtrack or admit they were wrong. They put the cart before the horse, now desperate to find any results that match their hypothesis.",
]
query = "I will never understand the decision making of the people of Alabama. Their new Senator is a definite downgrade. You have served with honor.  Well done."


class Test(unittest.TestCase):
    """ Test """

    def test(self):
        model = tweetnlp.load_model('sentence_embedding')
        vectors = model.predict(tweet_corpus, batch_size=3)
        print(vectors.shape)
        sims = []
        for i in tweet_corpus:
            sims.append(model.similarity(query, i))
        print(sims)


if __name__ == "__main__":
    unittest.main()
