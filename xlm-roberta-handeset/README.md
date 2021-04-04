# XLM-RoBERTa HanDeSeT

Scripts for training XLM-RoBERTa on HanDeSeT for sentiment classification.
XLM-RoBERTa Large was not used due to long sentence length in the dataset,
resulting in difficulties fitting model on a TITAN V 12GB GPU, even with a
batch size of 2.

Run `python train.py -h` for list of arguments.

## xlm-roberta-base-sst-2-handeset

XLM-RoBERTa Base model first finetuned on GLUE SST-2, then further finetuned
on HanDeSeT for 100 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 2e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 3e-5
Epoch: 38

Accuracy: 0.713
F1: 0.735
Precision: 0.739
Recall: 0.731
```

https://drive.google.com/file/d/17m6rMWl4Fge-svjJgngS8blltdcMx1Tj/view?usp=sharing

Training command:

```
python train.py --handeset_csv_path=HanDeSeT.csv --model_name_or_dir=xlm-roberta-base-sst-2 --output_dir=outputs --logging_dir=logs --batch_size=8 --gradient_accumulation_steps=4  --learning_rate=3e-5
```

## xlm-roberta-base-handeset

XLM-RoBERTa Base model finetuned on HanDeSeT for 100 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 2e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 3e-5
Epoch: 27

Accuracy: 0.734
F1: 0.768
Precision: 0.730
Recall: 0.810
```

https://drive.google.com/file/d/1fWBOztniQJwaw04Aj2cS5Hzo-Gis5lG0/view?usp=sharing

Training command:

```
python train.py --handeset_csv_path=HanDeSeT.csv --model_name_or_dir=xlm-roberta-base --output_dir=outputs --logging_dir=logs --batch_size=8 --gradient_accumulation_steps=4  --learning_rate=3e-5
```
