# XLM-RoBERTa Singapore Hansard NER

Scripts for training XLM-RoBERTa on Singapore Hansard NER.

Run `python train.py -h` for list of arguments.

## Singapore Hansard NER Dataset

Singapore Hansard NER Dataset was randomly split 80/20 to create
training and valiation sets.

TODO: Dataset statistics

[xlm-roberta-base-ontonotes5-sh-ner.tar.xz](https://drive.google.com/file/d/1B9Lqb3hlMQc9zCmN-Tgob96Eoy2ghgdD/view?usp=sharing)

## xlm-roberta-base-sh-ner

XLM-RoBERTa Base model finetuned on Singapore Hansard NER for 30 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 3e-5}.
Model was evaluated on the validation set after each epoch.
Best performing model:

```
Learning Rate: 3e-5
Epoch: 25

Accuracy: 0.974
F1: 0.786
Precision: 0.742
Recall: 0.837
```

Training command:

```
python train.py --train_json_path=sh_ner_train.json --val_json_path=sh_ner_val.json --model_name_or_dir=xlm-roberta-base --output_dir=outputs --logging_dir=logs --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=3e-5
```

## xlm-roberta-base-ontonotes5-sh-ner

XLM-RoBERTa Base model finetuned on OntoNotes 5, then further finetuned
on Singapore Hansard NER for 30 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 3e-5}.
Model was evaluated on the validation set after each epoch.
Best performing model:

```
Learning Rate: 1e-5
Epoch: 24

Accuracy: 0.977
F1: 0.819
Precision: 0.778
Recall: 0.864
```

Training command:

```
python train.py --train_json_path=sh_ner_train.json --val_json_path=sh_ner_val.json --model_name_or_dir=asahi417/tner-xlm-roberta-base-ontonotes5 --output_dir=outputs --logging_dir=logs --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=1e-5
```
