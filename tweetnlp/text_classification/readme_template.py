import json
import os
import shutil
from os.path import join as pj
from itertools import chain

sample = {
    "topic_classification": ['Get the all-analog Classic Vinyl Edition of "Takin Off" Album from {@herbiehancock@} via {@bluenoterecords@} link below {{URL}}'],
    'sentiment': ["Yes, including Medicare and social security saving👍"],
    'offensive': ["All two of them taste like ass."],
    'irony': ['If you wanna look like a badass, have drama on social media'],
    'hate': ['Whoever just unfollowed me you a bitch'],
    'emotion': ['I love swimming for the same reason I love meditating...the feeling of weightlessness.'],
    'emoji': ['Beautiful sunset last night from the pontoon @TupperLakeNY']
}

bib = """
```
@inproceedings{camacho-collados-etal-2022-tweetnlp,
    title={{T}weet{NLP}: {C}utting-{E}dge {N}atural {L}anguage {P}rocessing for {S}ocial {M}edia},
    author={Camacho-Collados, Jose and Rezaee, Kiamehr and Riahi, Talayeh and Ushio, Asahi and Loureiro, Daniel and Antypas, Dimosthenis and Boisson, Joanne and Espinosa-Anke, Luis and Liu, Fangyu and Mart{\'\i}nez-C{\'a}mara, Eugenio and others},
    author = "Ushio, Asahi  and
      Camacho-Collados, Jose",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2022",
    address = "Abu Dhabi, U.A.E.",
    publisher = "Association for Computational Linguistics",
}
```
"""


def get_readme(model_name: str,
               metric_file: str,
               dataset_name: str,
               dataset_type: str,
               language_model: str,
               split_test: str,
               split_train: str,
               split_validation: str,
               widget_sample_sentence: str = None,
               widget_type: str = None):
    if widget_type is None and widget_sample_sentence is None:
        widgets = [[f'- text: {_v}\n  example_title: "{k} {n + 1}" ' for n, _v in enumerate(v)] for k, v in sample.items()]
        widgets_str = '\n'.join(list(chain(*widgets)))
    elif widget_type is not None:
        widgets_str = f'- text: {sample[widget_type]}\n  example_title: "{widget_type}"'
    else:
        widgets_str = f'- text: {widget_sample_sentence}\n  example_title: "Example"'

    metric_file_model = pj(os.path.basename(model_name), os.path.basename(metric_file))
    evaluation_result = None
    if os.path.exists(metric_file):
        shutil.copy2(metric_file, os.path.basename(model_name))
    if os.path.exists(metric_file_model):
        with open(metric_file_model) as f:
            evaluation_result = json.load(f)
    return f"""---
datasets:
- {dataset_type}
metrics:
- f1
- accuracy
model-index:
- name: {model_name}
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: {dataset_type}
      type: {dataset_type if dataset_name is None else dataset_name}
      split: {split_test} 
    metrics:
    - name: Micro F1 ({dataset_type}{f"/{dataset_name}" if dataset_name is not None else ""})
      type: micro_f1_{dataset_type}{f"/{dataset_name}" if dataset_name is not None else ""}
      value: {evaluation_result[f'eval_f1'] if evaluation_result is not None else None}
    - name: Macro F1 ({dataset_type}{f"/{dataset_name}" if dataset_name is not None else ""})
      type: micro_f1_{dataset_type}{f"/{dataset_name}" if dataset_name is not None else ""}
      value: {evaluation_result[f'eval_f1_macro'] if evaluation_result is not None else None}
    - name: Accuracy ({dataset_type}{f"/{dataset_name}" if dataset_name is not None else ""})
      type: accuracy_{dataset_type}{f"/{dataset_name}" if dataset_name is not None else ""}
      value: {evaluation_result[f'eval_accuracy'] if evaluation_result is not None else None}
pipeline_tag: text-classification
widget:
{widgets_str}
---
# {model_name} 

This model is a fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) on the 
[`{dataset_type}{f" ({dataset_name})" if dataset_name is not None else ""}`](https://huggingface.co/datasets/{dataset_type}) 
via [`tweetnlp`](https://github.com/cardiffnlp/tweetnlp).
Training split is `{split_train}` and parameters have been tuned on the validation split `{split_validation}`.

Following metrics are achieved on the test split `{split_test}` ([link](https://huggingface.co/{model_name}/raw/main/{os.path.basename(metric_file)})).

- F1 (micro): {evaluation_result[f'eval_f1'] if evaluation_result is not None else None}
- F1 (macro): {evaluation_result[f'eval_f1_macro'] if evaluation_result is not None else None}
- Accuracy: {evaluation_result[f'eval_accuracy'] if evaluation_result is not None else None}

### Usage
Install tweetnlp via pip.
```shell
pip install tweetnlp
```
Load the model in python.
```python
import tweetnlp
model = tweetnlp.Classifier("{model_name}", max_length=128)
model.predict('{sample['topic_classification'][0] if widget_sample_sentence is None else widget_sample_sentence}')
```

### Reference
{bib}

"""
