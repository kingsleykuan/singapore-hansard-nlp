import json

from torch.utils.data import Dataset

def create_sh_sentiment_dataset(train_json_path, val_json_path, tokenizer):
    sentences_train = []
    labels_train = []

    sentences_val = []
    labels_val = []

    with open(train_json_path) as json_file:
        data = json.load(json_file)
    for pair in data:
        sentences_train.append(pair['text'])
        labels_train.append(pair['sentiment'])

    with open(val_json_path) as json_file:
        data = json.load(json_file)
    for pair in data:
        sentences_val.append(pair['text'])
        labels_val.append(pair['sentiment'])

    sh_sentiment_train_dataset = SingaporeHansardSentimentDataset(sentences_train, labels_train, tokenizer)
    sh_sentiment_val_dataset = SingaporeHansardSentimentDataset(sentences_val, labels_val, tokenizer)

    return sh_sentiment_train_dataset, sh_sentiment_val_dataset

class SingaporeHansardSentimentDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.encodings = tokenizer(self.sentences, padding=False, truncation=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.encodings.items()}
        item['label'] = self.labels[idx]
        return item
