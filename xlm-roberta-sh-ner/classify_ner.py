import argparse
import json
import os

import torch
from seqeval import metrics
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast

from sh_ner_dataset import id_to_label

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# NOTE: Assumes that sentiment has already been classified and JSON file is updated

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Perform NER on Singapore Hansard using XLM-RoBERTa model')

    parser.add_argument('input_dir_path', type=str,
        help='Path of directroy to read JSON files from.')

    parser.add_argument('output_dir_path', type=str,
        help='Path of directroy to write JSON files to.')

    parser.add_argument('model_name_or_dir', type=str,
        help='Name or directory of model.')

    return parser

def main(input_dir_path, output_dir_path, model_name_or_dir):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name_or_dir)
    model = XLMRobertaForTokenClassification.from_pretrained(model_name_or_dir).to(DEVICE)
    model.eval()

    os.makedirs(output_dir_path, exist_ok=True)

    for file_name in os.listdir(input_dir_path):
        if file_name.endswith('.json'):
            count = 0
            input_file_path = os.path.join(input_dir_path, file_name)
            with open(input_file_path) as json_file:
                data = json.load(json_file)

            for session in data['sessions']:
                for speech in session['speeches']:
                    for text_sentiment in speech['content']:
                        text = text_sentiment['text']
                        inputs = tokenizer(
                            text,
                            padding=False,
                            truncation=True,
                            return_special_tokens_mask=True,
                            return_offsets_mapping=True)

                        with torch.no_grad():
                            input_ids = torch.tensor(inputs['input_ids']).unsqueeze(0).to(DEVICE)
                            attention_mask = torch.tensor(inputs['attention_mask']).unsqueeze(0).to(DEVICE)
                            outputs = model(input_ids, attention_mask).logits
                            predictions = torch.argmax(outputs, dim=2)[0].detach().cpu().numpy()

                        special_tokens_mask = inputs['special_tokens_mask']
                        offset_mapping = inputs['offset_mapping']

                        start_index = 0
                        end_index = 0
                        previous_iob_entity = None

                        entities = []

                        for i in range(len(predictions)):
                            if special_tokens_mask[i] == 0 and predictions[i] != 0:
                                iob_entity = id_to_label(predictions[i])

                                if iob_entity[:2] == 'B-' or previous_iob_entity is None or iob_entity[2:] != previous_iob_entity[2:]:
                                    if previous_iob_entity is not None:
                                        label = previous_iob_entity[2:]
                                        start = start_index
                                        end = end_index

                                        if text[start] == ' ':
                                            start += 1

                                        word = text[start:end]

                                        entities.append({
                                            'word': word,
                                            'start': start,
                                            'end': end,
                                            'label': label,
                                        })

                                    start_index = offset_mapping[i][0]

                                end_index = offset_mapping[i][1]
                                previous_iob_entity = iob_entity

                        if previous_iob_entity is not None:
                            label = previous_iob_entity[2:]
                            start = start_index
                            end = end_index

                            if text[start] == ' ':
                                start += 1

                            word = text[start:end]

                            entities.append({
                                'word': word,
                                'start': start,
                                'end': end,
                                'label': label,
                            })

                        text_sentiment['entities'] = entities
                        count += 1

            output_file_path = os.path.join(output_dir_path, file_name)
            with open(output_file_path, 'w') as json_file:
                json.dump(data, json_file)

            print("File: {}, Count: {}".format(file_name, count))

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args.input_dir_path, args.output_dir_path, args.model_name_or_dir)
