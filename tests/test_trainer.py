import logging
import shutil
import tweetnlp
from pprint import pprint

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
sample = [
    "How many more days until opening day? 😩"
    "All two of them taste like ass.",
    "If you wanna look like a badass, have drama on social media",
    "Whoever just unfollowed me you a bitch",
    "I love swimming for the same reason I love meditating...the feeling of weightlessness.",
    "Beautiful sunset last night from the pontoon @ Tupper Lake, New York",
    'Jacob Collier is a Grammy-awarded English artist from London.'
]
language_model = 'cardiffnlp/twitter-roberta-base-dec2021'
task = "irony"

dataset, label_to_id = tweetnlp.load_dataset(task)
trainer_class = tweetnlp.load_trainer(task)
trainer = trainer_class(
    language_model=language_model,
    dataset=dataset,
    label_to_id=label_to_id,
    max_length=128,
    split_train='train',
    split_test='test',
    output_dir=f'model_ckpt/test'
)
trainer.train(down_sample_size_train=1000, ray_result_dir="ray_results/test")
trainer.save_model()
trainer.evaluate()
trainer.push_to_hub(hf_organization='cardiffnlp', model_alias='test')

# sample prediction
output = trainer.predict(sample)
pprint(f"Sample Prediction: {language_model} ({task})")
for s, p in zip(sample, output):
    pprint(s)
    pprint(p)

# clean up logs
shutil.rmtree(f'model_ckpt/test')
shutil.rmtree("test")



