import json
import random

from sklearn.model_selection import train_test_split

JSON_FILE_PATHS = [
    'ApostilleBill_ner.json',
    'ClimateChange_ner.json',
    'Dialogue&DissentProgYale-NUS_ner.json',
    'SingaporesJusticeSystem_ner.json'
]

COMBINED_JSON_PATH = 'sh_ner.json'
TRAIN_JSON_PATH = 'sh_ner_train.json'
VAL_JSON_PATH = 'sh_ner_val.json'

SPLIT = 0.2
RANDOM_SEED = 42

def main(json_file_paths, combined_json_path, train_json_path, val_json_path, split, random_seed=42):
    data = []

    for json_file_path in json_file_paths:
        with open(json_file_path) as json_file:
            data += json.load(json_file)

    train, val = train_test_split(data, test_size=split, random_state=random_seed)

    with open(combined_json_path, 'w') as json_file:
        json.dump(data, json_file)

    with open(train_json_path, 'w') as json_file:
        json.dump(train, json_file)

    with open(val_json_path, 'w') as json_file:
        json.dump(val, json_file)

if __name__ == '__main__':
    main(JSON_FILE_PATHS, COMBINED_JSON_PATH, TRAIN_JSON_PATH, VAL_JSON_PATH, SPLIT, RANDOM_SEED)
