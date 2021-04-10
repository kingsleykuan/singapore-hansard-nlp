import argparse
import json
import os

import sklearn.metrics as metrics
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Test sentiment classification on Singapore Hansard using XLM-RoBERTa model')

    parser.add_argument('json_path', type=str,
        help='Path of JSON file containing Singapore Hansard data.')

    parser.add_argument('model_name_or_dir', type=str,
        help='Name or directory of model.')

    return parser

def main(
        json_path,
        model_name_or_dir):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name_or_dir)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_name_or_dir).to(DEVICE)
    model.eval()

    with open(json_path) as json_file:
        data = json.load(json_file)

    predictions = []
    labels = []

    for pair in data:
        sentence = pair['text']
        label = pair['sentiment']

        inputs = tokenizer.encode(
            sentence, padding=False, truncation=True, return_tensors='pt').to(DEVICE)

        with torch.no_grad():
            output = model(inputs).logits
            prediction = torch.argmax(output, dim=-1)[0].item()

        predictions.append(prediction)
        labels.append(label)
    
    print(metrics.classification_report(labels, predictions, digits=6))

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(
        args.json_path,
        args.model_name_or_dir)
