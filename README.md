# singapore-hansard-nlp
Singapore Hansard NLP

[singapore-hansard-sentiment-ner-final.zip](https://drive.google.com/file/d/1xWDplG7ythfnv3EZjxhjEmtUHwS9st-E/view?usp=sharing)

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

[xlm-roberta-base-sst-2-sh-sentiment.tar.xz](https://drive.google.com/file/d/1toqvkwWjXuHH0EIHHjJv9V9x5FZ-0Pba/view?usp=sharing)

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

## Named Entity Recognition (NER)

Models were evaluated on Singapore Hansard NER Dataset validation set.

| Model                                     | F1 Score | Precision | Recall |
|-------------------------------------------|----------|-----------|--------|
| asahi417/tner-xlm-roberta-base-ontonotes5 | 0.343    | 0.274     | 0.458  |
| xlm-roberta-base-sh-ner                   | 0.786    | 0.742     | 0.837  |
| xlm-roberta-base-ontonotes5-sh-ner        | **0.819**| **0.778** | **0.864** |

[xlm-roberta-base-ontonotes5-sh-ner.tar.xz](https://drive.google.com/file/d/1B9Lqb3hlMQc9zCmN-Tgob96Eoy2ghgdD/view?usp=sharing)

```
python test_ner.py sh_ner_val.json models/xlm-roberta-base-ontonotes5-sh-ner
```



asahi417/tner-xlm-roberta-base-ontonotes5
```
Accuracy: 0.9153978551568913
F1: 0.3427762039660056
Precision: 0.2737556561085973
Recall: 0.4583333333333333

                 precision    recall  f1-score   support

CARDINAL NUMBER   0.000000  0.000000  0.000000         0
           DATE   0.414634  0.680000  0.515152        25
          EVENT   0.000000  0.000000  0.000000         0
            FAC   0.000000  0.000000  0.000000         1
            GPE   0.812500  0.672414  0.735849        58
            LAW   0.464286  0.406250  0.433333        32
            LOC   0.000000  0.000000  0.000000         0
          MONEY   0.000000  0.000000  0.000000         0
           NORP   0.000000  0.000000  0.000000        16
 ORDINAL NUMBER   0.000000  0.000000  0.000000         0
            ORG   0.337931  0.620253  0.437500        79
        PERCENT   0.000000  0.000000  0.000000         0
         PERSON   0.036585  0.056604  0.044444        53
       QUANTITY   0.000000  0.000000  0.000000         0
           TIME   0.000000  0.000000  0.000000         0
    WORK_OF_ART   0.000000  0.000000  0.000000         0

      micro avg   0.273756  0.458333  0.342776       264
      macro avg   0.129121  0.152220  0.135392       264
   weighted avg   0.382514  0.458333  0.402813       264
```

xlm-roberta-base-sh-ner
```
Accuracy: 0.974050046339203
F1: 0.7864768683274023
Precision: 0.7416107382550335
Recall: 0.8371212121212122

              precision    recall  f1-score   support

        DATE   0.681818  0.600000  0.638298        25
         FAC   0.000000  0.000000  0.000000         1
         GPE   0.838710  0.896552  0.866667        58
         LAW   0.676471  0.718750  0.696970        32
        NORP   0.736842  0.875000  0.800000        16
         ORG   0.620370  0.848101  0.716578        79
      PERSON   0.943396  0.943396  0.943396        53

   micro avg   0.741611  0.837121  0.786477       264
   macro avg   0.642515  0.697400  0.665987       264
weighted avg   0.750517  0.837121  0.787639       264
```

xlm-roberta-base-ontonotes5-sh-ner
```
Accuracy: 0.9770951939626639
F1: 0.8186714542190305
Precision: 0.7781569965870307
Recall: 0.8636363636363636

              precision    recall  f1-score   support

        DATE   0.739130  0.680000  0.708333        25
         FAC   0.000000  0.000000  0.000000         1
         GPE   0.910714  0.879310  0.894737        58
         LAW   0.774194  0.750000  0.761905        32
        NORP   0.882353  0.937500  0.909091        16
         ORG   0.633929  0.898734  0.743455        79
      PERSON   0.925926  0.943396  0.934579        53

   micro avg   0.778157  0.863636  0.818671       264
   macro avg   0.695178  0.726992  0.707443       264
weighted avg   0.792977  0.863636  0.821194       264
```
