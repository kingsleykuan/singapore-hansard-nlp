import csv

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def create_handeset_dataset(handeset_csv_path, tokenizer, test_size=0.2, random_state=42):
    sentences = []
    labels = []

    with open(handeset_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for i in range(1, 6):
                utterance = row['utt' + str(i)]
                manual_speech = int(row['manual speech'])

                if utterance:
                    sentences.append(utterance)
                    labels.append(manual_speech)
    
    sentences_train, sentences_val, labels_train, labels_val = train_test_split(
        sentences, labels, test_size=0.2, random_state=42)
    
    handeset_train_dataset = HandesetDataset(sentences_train, labels_train, tokenizer)
    handeset_val_dataset = HandesetDataset(sentences_val, labels_val, tokenizer)

    return handeset_train_dataset, handeset_val_dataset

class HandesetDataset(Dataset):
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
