# singapore-hansard-nlp
Singapore Hansard NLP

Note that 28 JSON files in the 12th session from 2011-10-10 to 2012-08-13 are excluded
as they use an old format that is difficult to parse.

## Sentiment Analysis

Models were evaluated on Singapore Hansard Sentiment Dataset validation set.

| Model                                        | Accuracy | F1 Score | Precision | Recall |
|----------------------------------------------|----------|----------|-----------|--------|
| xlm-roberta-base-sst-2                       | 0.780    | 0.834    | 0.820     | 0.849  |
| xlm-roberta-base-handeset                    | 0.447    | 0.547    | 0.587     | 0.512  |
| xlm-roberta-base-sst-2-handeset              | 0.561    | 0.691    | 0.637     | 0.756  |
| xlm-roberta-base-sh-sentiment                | 0.856    | 0.889    | 0.894     | **0.884**  |
| xlm-roberta-base-sst-2-sh-sentiment          | **0.879**| **0.904**| **0.938** | 0.872  |
| xlm-roberta-base-handeset-sh-sentiment       | 0.773    | 0.828    | 0.818     | 0.837  |
| xlm-roberta-base-sst-2-handeset-sh-sentiment | 0.841    | 0.873    | 0.911     | 0.837  |

```
python test_sentiment.py input.json models/xlm-roberta-base-sst-2-sh-sentiment
```

[singapore-hansard-sentiment-final.zip](https://drive.google.com/file/d/14yZRPLvQ7usliO1WOdFKmqoWgtJjax0C/view?usp=sharing)

### Raw Results

xlm-roberta-base-sst-2
```
              precision    recall  f1-score   support

           0   0.697674  0.652174  0.674157        46
           1   0.820225  0.848837  0.834286        86

    accuracy                       0.780303       132
   macro avg   0.758950  0.750506  0.754222       132
weighted avg   0.777518  0.780303  0.778483       132
```

xlm-roberta-base-handeset
```
              precision    recall  f1-score   support

           0   0.263158  0.326087  0.291262        46
           1   0.586667  0.511628  0.546584        86

    accuracy                       0.446970       132
   macro avg   0.424912  0.418857  0.418923       132
weighted avg   0.473929  0.446970  0.457608       132
```

xlm-roberta-base-sst-2-handeset
```
              precision    recall  f1-score   support

           0   0.300000  0.195652  0.236842        46
           1   0.637255  0.755814  0.691489        86

    accuracy                       0.560606       132
   macro avg   0.468627  0.475733  0.464166       132
weighted avg   0.519727  0.560606  0.533052       132
```

xlm-roberta-base-sh-sentiment
```
              precision    recall  f1-score   support

           0   0.787234  0.804348  0.795699        46
           1   0.894118  0.883721  0.888889        86

    accuracy                       0.856061       132
   macro avg   0.840676  0.844034  0.842294       132
weighted avg   0.856870  0.856061  0.856414       132

```

xlm-roberta-base-sst-2-sh-sentiment
```
              precision    recall  f1-score   support

           0   0.788462  0.891304  0.836735        46
           1   0.937500  0.872093  0.903614        86

    accuracy                       0.878788       132
   macro avg   0.862981  0.881699  0.870175       132
weighted avg   0.885562  0.878788  0.880308       132
```

xlm-roberta-base-handeset-sh-sentiment
```
              precision    recall  f1-score   support

           0   0.681818  0.652174  0.666667        46
           1   0.818182  0.837209  0.827586        86

    accuracy                       0.772727       132
   macro avg   0.750000  0.744692  0.747126       132
weighted avg   0.770661  0.772727  0.771508       132
```

xlm-roberta-base-sst-2-handeset-sh-sentiment
```
              precision    recall  f1-score   support

           0   0.735849  0.847826  0.787879        46
           1   0.911392  0.837209  0.872727        86

    accuracy                       0.840909       132
   macro avg   0.823621  0.842518  0.830303       132
weighted avg   0.850218  0.840909  0.843159       132
```
