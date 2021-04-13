import argparse

import torch
from seqeval import metrics
from transformers import XLMRobertaForTokenClassification, XLMRobertaTokenizerFast

from sh_ner_dataset import create_sh_ner_dataset, id_to_label

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Test NER on Singapore Hansard using XLM-RoBERTa model')

    parser.add_argument('json_path', type=str,
        help='Path of JSON file containing Singapore Hansard data.')

    parser.add_argument('model_name_or_dir', type=str,
        help='Name or directory of model.')

    return parser

def main(json_path, model_name_or_dir):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name_or_dir)
    model = XLMRobertaForTokenClassification.from_pretrained(model_name_or_dir).to(DEVICE)
    model.eval()

    _, sh_ner_val_dataset = create_sh_ner_dataset(json_path, json_path, tokenizer)

    filtered_labels = []
    filtered_preds = []

    for instance in sh_ner_val_dataset:
        with torch.no_grad():
            input_ids = torch.tensor(instance['input_ids']).unsqueeze(0).to(DEVICE)
            attention_mask = torch.tensor(instance['attention_mask']).unsqueeze(0).to(DEVICE)
            outputs = model(input_ids, attention_mask).logits
            predictions = torch.argmax(outputs, dim=2)[0].detach().cpu().numpy()

            filtered_labels_inner = []
            filtered_preds_inner = []

            for label, prediction in zip(instance['labels'], predictions):
                if label != -100:
                    filtered_labels_inner.append(id_to_label(label))
                    filtered_preds_inner.append(id_to_label(prediction))

            filtered_labels.append(filtered_labels_inner)
            filtered_preds.append(filtered_preds_inner)

    accuracy = metrics.accuracy_score(filtered_labels, filtered_preds)
    f1 = metrics.f1_score(filtered_labels, filtered_preds)
    precision = metrics.precision_score(filtered_labels, filtered_preds)
    recall = metrics.recall_score(filtered_labels, filtered_preds)

    print("Accuracy: {}".format(accuracy))
    print("F1: {}".format(f1))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))

    print(metrics.classification_report(filtered_labels, filtered_preds, digits=6))

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(args.json_path, args.model_name_or_dir)
