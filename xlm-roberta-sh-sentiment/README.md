# XLM-RoBERTa Singapore Hansard Sentiment

Scripts for training XLM-RoBERTa on Singapore Hansard Sentiment for 
sentiment classification.

Run `python train.py -h` for list of arguments.


## Singapore Hansard Sentiment Dataset

Singapore Hansard Sentiment Dataset was randomly split 80/20 to create
training and valiation sets.

|                    | Total | Training | Validation |
|--------------------|-------|----------|------------|
| Total Instances    | 658   | 526      | 132        |
| Positive Instances | 441   | 355      | 86         |
| Negative Instances | 217   | 171      | 46         |

Due to the class inbalance, oversampling was performed on the training
set, resulting in 710 instances in total.

## xlm-roberta-base-sh-sentiment

XLM-RoBERTa Base model finetuned on Singapore Hansard Sentiment for 10 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 7e-6
Epoch: 10

Accuracy: 0.856
F1: 0.888
Precision: 0.894
Recall: 0.884
```

Training command:

```
python train.py --train_json_path=sh_sentiment_train.json --val_json_path=sh_sentiment_val.json --model_name_or_dir=xlm-roberta-base --output_dir=outputs --logging_dir=logs --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=7e-6
```

## xlm-roberta-base-sst-2-sh-sentiment

XLM-RoBERTa Base model first finetuned on GLUE SST-2, then further finetuned
on Singapore Hansard Sentiment for 10 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 1e-5
Epoch: 2

Accuracy: 0.879
F1: 0.904
Precision: 0.938
Recall: 0.872
```

Training command:

```
python train.py --train_json_path=sh_sentiment_train.json --val_json_path=sh_sentiment_val.json --model_name_or_dir=xlm-roberta-base-sst-2 --output_dir=outputs --logging_dir=logs --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=1e-5
```

## xlm-roberta-base-handeset-sh-sentiment

XLM-RoBERTa Base model first finetuned on HanDeSeT, then further finetuned
on Singapore Hansard Sentiment for 10 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 3e-5
Epoch: 10

Accuracy: 0.773
F1: 0.828
Precision: 0.818
Recall: 0.837
```

Training command:

```
python train.py --train_json_path=sh_sentiment_train.json --val_json_path=sh_sentiment_val.json --model_name_or_dir=xlm-roberta-base-handeset --output_dir=outputs --logging_dir=logs --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=3e-5
```

## xlm-roberta-base-sst-2-handeset-sh-sentiment

XLM-RoBERTa Base model first finetuned on GLUE SST-2, then HanDeSeT, then
on Singapore Hansard Sentiment for 10 epochs.
Hyperparameter search was done for learning rate in {7e-6, 1e-5, 3e-5}.
Model was evaluated on the dev set after each epoch. Best performing model:

```
Learning Rate: 3e-5
Epoch: 2

Accuracy: 0.841
F1: 0.873
Precision: 0.911
Recall: 0.837
```

Training command:

```
python train.py --train_json_path=sh_sentiment_train.json --val_json_path=sh_sentiment_val.json --model_name_or_dir=xlm-roberta-base-sst-2-handeset --output_dir=outputs --logging_dir=logs --batch_size=16 --gradient_accumulation_steps=2 --learning_rate=3e-5
```
