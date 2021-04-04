# XLM-RoBERTa SST-2

Scripts for training XLM-RoBERTa on GLUE SST-2 (Stanford Sentiment Treebank) for sentiment classification.

Run `python train.py -h` for list of arguments.

## xlm-roberta-base-sst-2

XLM-RoBERTa Base model trained on GLUE SST-2 trained for 10 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 2e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 7e-6
Epoch: 5

Accuracy: 0.929
F1: 0.932
Precision: 0.906
Recall: 0.959
```

https://drive.google.com/file/d/101-X7Jo3-16xhx77Q3BsjCVX_GM4EqWQ/view?usp=sharing

Training command:

```
python train.py --output_dir=outputs --logging_dir=logs --batch_size=32 --gradient_accumulation_steps=1 --learning_rate=7e-6
```

## xlm-roberta-large-sst-2

XLM-RoBERTa Large model trained on GLUE SST-2 trained for 10 epochs.
Hyperparameter search was done for learning rate in {5e-6, 7e-6, 1e-5, 2e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 1e-5
Epoch: 4

Accuracy: 0.947
F1: 0.949
Precision: 0.936
Recall: 0.962
```

https://drive.google.com/file/d/1scBLI5Dd3jHBbMLSqhK7GO53ZPoeUkIN/view?usp=sharing

Training command:

```
python train.py --output_dir=outputs --logging_dir=logs --large --batch_size=8 --gradient_accumulation_steps=4 --learning_rate=1e-5
```
