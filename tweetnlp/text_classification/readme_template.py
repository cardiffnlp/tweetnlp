import json
import os
import shutil
from os.path import join as pj

sample = 'Get the all-analog Classic Vinyl Edition of "Takin\' Off" Album from {@herbiehancock@} via {@bluenoterecords@} link below: {{URL}}'
bib = """
@inproceedings{dimosthenis-etal-2022-twitter,
    title = "{T}witter {T}opic {C}lassification",
    author = "Antypas, Dimosthenis  and
    Ushio, Asahi  and
    Camacho-Collados, Jose  and
    Neves, Leonardo  and
    Silva, Vitor  and
    Barbieri, Francesco",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics"
}
"""


def get_readme(model_name: str,
               metric_file: str,
               dataset_name: str,
               dataset_type: str,
               language_model: str,
               split_test: str,
               split_train: str,
               split_validation: str,
               widget_sample_sentence: str = None):
    widget_sample_sentence = sample if widget_sample_sentence is None else widget_sample_sentence
    evaluation_result = None
    if os.path.exists(metric_file):
        shutil.copy2(metric_file, model_name)
        metric_file = pj(model_name, os.path.basename(metric_file))
        with open(metric_file) as f:
            evaluation_result = json.load(f)
    return f"""---
datasets:
- {dataset_name}
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
      name: {dataset_name}
      type: {'default' if dataset_type is None else dataset_type}
      split: {split_test} 
    metrics:
    - name: F1
      type: f1
      value: {evaluation_result[f'{split_test}/eval_f1'] if evaluation_result is not None else None}
    - name: F1 (macro)
      type: f1_macro
      value: {evaluation_result[f'{split_test}/eval_f1_macro'] if evaluation_result is not None else None}
    - name: Accuracy
      type: accuracy
      value: {evaluation_result[f'{split_test}/eval_accuracy'] if evaluation_result is not None else None}
pipeline_tag: text-classification
widget:
- text: {widget_sample_sentence}
  example_title: "Example"
---
# {model_name}

This model is a fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) on the 
[`{dataset_name}{f" ({dataset_type})" if dataset_type is not None else ""})`](https://huggingface.co/datasets/{dataset_name}) 
via [`tweetnlp`](https://github.com/cardiffnlp/tweetnlp).
Training split is `{split_train}` and parameters have been tuned on the validation split `{split_validation}`.

Following metrics are achieved on the test split `{split_test}` ([link](https://huggingface.co/{model_name}/raw/main/{os.path.basename(metric_file)})).

- F1 (micro): {evaluation_result[f'{split_test}/eval_f1'] if evaluation_result is not None else None}
- F1 (macro): {evaluation_result[f'{split_test}/eval_f1_macro'] if evaluation_result is not None else None}
- Accuracy: {evaluation_result[f'{split_test}/eval_accuracy'] if evaluation_result is not None else None}

### Usage
Install tweetnlp via pip.
```shell
pip install tweetnlp
```
Load the model in python.
```python
import tweetnlp
model = tweetnlp.Classifier({model_name}, max_length=128)
model.predict({widget_sample_sentence})
```

### Reference

```
{bib}
```
"""
