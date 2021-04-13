import argparse

from seqeval import metrics
from transformers import (DataCollatorForTokenClassification, Trainer, TrainingArguments,
    XLMRobertaForTokenClassification, XLMRobertaTokenizerFast)

from sh_ner_dataset import create_sh_ner_dataset, id_to_label

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Train XLM-RoBERTa model on Singapore Hansard for NER.')

    parser.add_argument('--train_json_path', default='sh_ner_train.json', type=str,
        help='Path to Singapore Hansard NER training JSON file.')

    parser.add_argument('--val_json_path', default='sh_ner_val.json', type=str,
        help='Path to Singapore Hansard NER validation JSON file.')

    parser.add_argument('--model_name_or_dir', default='model', type=str,
        help='Name or directory of model to finetune.')

    parser.add_argument('--output_dir', default='outputs', type=str,
        help='Directory to output trained models.')

    parser.add_argument('--logging_dir', default='logs', type=str,
        help='Directory to output TensorBoard logs.')

    parser.add_argument('--logging_steps', default=2, type=int,
        help='Number of steps between logging.')

    parser.add_argument('--batch_size', default=16, type=int,
        help='Batch size during training and evaluation.')

    parser.add_argument('--gradient_accumulation_steps', default=2, type=int,
        help='Number of steps to accumulate gradients for. Effectively scales batch size.')

    parser.add_argument('--learning_rate', default=1e-5, type=float,
        help='Learning rate to start at after warmup.')

    parser.add_argument('--num_train_epochs', default=30.0, type=float,
        help='Number of epochs to train for.')

    parser.add_argument('--warmup_ratio', default=1.0/30.0, type=float,
        help='Ratio of training to warmup, linearly increases learning rate from 0 to starting.')

    parser.add_argument('--num_classes', default=37, type=int,
        help='Number of classes.')

    return parser

def main(
        train_json_path,
        val_json_path,
        model_name_or_dir,
        output_dir,
        logging_dir,
        logging_steps,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        warmup_ratio,
        num_classes):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name_or_dir)
    model = XLMRobertaForTokenClassification.from_pretrained(model_name_or_dir, num_labels=num_classes)

    sh_ner_train_dataset, sh_ner_val_dataset = create_sh_ner_dataset(
        train_json_path, val_json_path, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        do_predict=False,
        evaluation_strategy='epoch',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        logging_dir=logging_dir,
        logging_strategy='steps',
        logging_steps=logging_steps,
        save_strategy='epoch',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sh_ner_train_dataset,
        eval_dataset=sh_ner_val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    filtered_labels = []
    filtered_preds = []

    for i in range(labels.shape[0]):
        filtered_labels_inner = []
        filtered_preds_inner = []

        for j in range(labels.shape[1]):


            if labels[i][j] != -100:
                filtered_labels_inner.append(id_to_label(labels[i][j]))
                filtered_preds_inner.append(id_to_label(preds[i][j]))

        filtered_labels.append(filtered_labels_inner)
        filtered_preds.append(filtered_preds_inner)

    return {
        'accuracy': metrics.accuracy_score(filtered_labels, filtered_preds),
        'f1': metrics.f1_score(filtered_labels, filtered_preds),
        'precision': metrics.precision_score(filtered_labels, filtered_preds),
        'recall': metrics.recall_score(filtered_labels, filtered_preds),
    }

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(
        args.train_json_path,
        args.val_json_path,
        args.model_name_or_dir,
        args.output_dir,
        args.logging_dir,
        args.logging_steps,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.num_train_epochs,
        args.warmup_ratio,
        args.num_classes)
