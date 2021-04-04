# singapore-hansard-nlp
Singapore Hansard NLP

## Sentiment Analysis

Sentiment was classified using `classify_sentiment.py` with various pretrained
models:

* `xlm-roberta-base-sst-2`
* `xlm-roberta-base-sst-2-handeset`
* `xlm-roberta-base-handeset`

```
python classify.py input_dir output_dir models/xlm-roberta-base-sst-2
```

Output:
https://drive.google.com/file/d/196quEqxtojy_gkBhNcXFfhEt8-4LOR0K/view?usp=sharing
