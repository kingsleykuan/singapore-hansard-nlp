import json

from torch.utils.data import Dataset

ID_TO_LABEL = {
    0: 'O',
    1: 'B-CARDINAL NUMBER',
    2: 'B-DATE',
    3: 'I-DATE',
    4: 'B-PERSON',
    5: 'I-PERSON',
    6: 'B-NORP',
    7: 'B-GPE',
    8: 'I-GPE',
    9: 'B-LAW',
    10: 'I-LAW',
    11: 'B-ORG',
    12: 'I-ORG',
    13: 'B-PERCENT',
    14: 'I-PERCENT',
    15: 'B-ORDINAL NUMBER',
    16: 'B-MONEY',
    17: 'I-MONEY',
    18: 'B-WORK_OF_ART',
    19: 'I-WORK_OF_ART',
    20: 'B-FAC',
    21: 'B-TIME',
    22: 'I-CARDINAL NUMBER',
    23: 'B-LOC',
    24: 'B-QUANTITY',
    25: 'I-QUANTITY',
    26: 'I-NORP',
    27: 'I-LOC',
    28: 'B-PRODUCT',
    29: 'I-TIME',
    30: 'B-EVENT',
    31: 'I-EVENT',
    32: 'I-FAC',
    33: 'B-LANGUAGE',
    34: 'I-PRODUCT',
    35: 'I-ORDINAL NUMBER',
    36: 'I-LANGUAGE',
    -100: 'MASK',
}

LABEL_TO_ID = {v: k for k, v in ID_TO_LABEL.items()}

def id_to_label(id):
    return ID_TO_LABEL[id]

def label_to_id(label):
    return LABEL_TO_ID[label]

def create_sh_ner_dataset(train_json_path, val_json_path, tokenizer):
    sentences_train = []
    sentences_entities_train = []

    sentences_val = []
    sentences_entities_val = []

    with open(train_json_path) as json_file:
        data = json.load(json_file)
    for sentence in data:
        sentences_train.append(sentence['content'])
        sentences_entities_train.append(sentence['entities'])

    with open(val_json_path) as json_file:
        data = json.load(json_file)
    for sentence in data:
        sentences_val.append(sentence['content'])
        sentences_entities_val.append(sentence['entities'])

    sh_ner_train_dataset = SingaporeHansardNerDataset(sentences_train, sentences_entities_train, tokenizer)
    sh_ner_val_dataset = SingaporeHansardNerDataset(sentences_val, sentences_entities_val, tokenizer)

    return sh_ner_train_dataset, sh_ner_val_dataset

class SingaporeHansardNerDataset(Dataset):
    def __init__(self, sentences, sentences_entities, tokenizer, trainer=True):
        self.sentences = sentences
        self.sentences_entities = sentences_entities

        self.id2label = ID_TO_LABEL
        self.label2id = LABEL_TO_ID

        encodings = []
        labels = []

        for sentence, entities in zip(sentences, sentences_entities):
            sentence_encodings = tokenizer(
                sentence,
                padding=False,
                truncation=False,
                return_special_tokens_mask=True,
                return_offsets_mapping=True)
            sentence_labels = sentence_encodings['input_ids'].copy()

            for i in range(len(sentence_labels)):
                if sentence_encodings['special_tokens_mask'][i] == 0:
                    sentence_labels[i] = 'O'

                    offset = sentence_encodings['offset_mapping'][i]
                    offset_range = set(range(offset[0], offset[1]))
                    for start, end, entity in entities:
                        if len(offset_range.intersection(range(start, end))) > 0:
                            beginning = 'B-' + entity
                            inside = 'I-' + entity

                            if offset[0] == start or (i > 0
                                    and sentence_labels[i-1] != beginning
                                    and sentence_labels[i-1] != inside):
                                sentence_labels[i] = beginning
                            else:
                                sentence_labels[i] = inside

                            break
                else:
                    sentence_labels[i] = 'MASK'

            if trainer:
                del sentence_encodings['special_tokens_mask']
                del sentence_encodings['offset_mapping']

            sentence_labels = [self.label2id[label] for label in sentence_labels]

            encodings.append(sentence_encodings)
            labels.append(sentence_labels)

        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item['labels'] = self.labels[idx]
        return item
