import json
import random

from sklearn.model_selection import train_test_split

JSON_FILE_PATHS = [
    'ApostilleBill_sentiment.json',
    'ClimateChange_sentiment.json',
    'Dialogue&DissentProgYale-NUS_sentiment.json',
    'SingaporesJusticeSystem_sentiment.json'
]

COMBINED_JSON_PATH = 'sh_sentiment.json'
TRAIN_JSON_PATH = 'sh_sentiment_train.json'
VAL_JSON_PATH = 'sh_sentiment_val.json'

SPLIT = 0.2
RANDOM_SEED = 42

def main(json_file_paths, combined_json_path, train_json_path, val_json_path, split, random_seed=42):
    data = []

    for json_file_path in json_file_paths:
        with open(json_file_path) as json_file:
            data += json.load(json_file)

    data = [pair for pair in data if isinstance(pair['sentiment'], int)]
    train, val = train_test_split(data, test_size=split, random_state=random_seed)

    positive = []
    negative = []
    for pair in train:
        if pair['sentiment'] == 0:
            negative.append(pair)
        elif pair['sentiment'] == 1:
            positive.append(pair)

    while len(negative) * 2 < len(positive):
        negative += negative

    diff = len(positive) - len(negative)

    random.seed(random_seed)
    sampled = random.sample(negative, diff)
    negative += sampled

    balanced_train = positive + negative
    random.shuffle(balanced_train)

    with open(combined_json_path, 'w') as json_file:
        json.dump(data, json_file)

    with open(train_json_path, 'w') as json_file:
        json.dump(balanced_train, json_file)

    with open(val_json_path, 'w') as json_file:
        json.dump(val, json_file)

if __name__ == '__main__':
    main(JSON_FILE_PATHS, COMBINED_JSON_PATH, TRAIN_JSON_PATH, VAL_JSON_PATH, SPLIT, RANDOM_SEED)
