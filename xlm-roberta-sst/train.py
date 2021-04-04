import argparse
from datasets import load_dataset
from sklearn import metrics
from transformers import (Trainer, TrainingArguments,
    XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast)

def create_arg_parser():
    parser = argparse.ArgumentParser(
        description='Train XLM-RoBERTa model on GLUE SST-2 for sentiment classification.')

    parser.add_argument('--output_dir', default='outputs', type=str,
        help='Directory to output trained models.')

    parser.add_argument('--logging_dir', default='logs', type=str,
        help='Directory to output TensorBoard logs.')

    parser.add_argument('--logging_steps', default=100, type=int,
        help='Number of steps between logging.')

    parser.add_argument('--large', action='store_true',
        help='Finetune xlm-roberta-large instead of xlm-roberta-base.')

    parser.add_argument('--batch_size', default=32, type=int,
        help='Batch size during training and evaluation.')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
        help='Number of steps to accumulate gradients for. Effectively scales batch size.')

    parser.add_argument('--learning_rate', default=7e-6, type=float,
        help='Learning rate to start at after warmup.')

    parser.add_argument('--num_train_epochs', default=10.0, type=float,
        help='Number of epochs to train for.')

    parser.add_argument('--warmup_ratio', default=1.0/10.0, type=float,
        help='Ratio of training to warmup, linearly increases learning rate from 0 to starting.')

    return parser

def main(
        output_dir,
        logging_dir,
        logging_steps,
        large,
        batch_size,
        gradient_accumulation_steps,
        learning_rate,
        num_train_epochs,
        warmup_ratio):
    sst_train_dataset = load_dataset('glue', 'sst2', split='train')
    sst_validation_dataset = load_dataset('glue', 'sst2', split='validation')

    if large:
        model_name = 'xlm-roberta-large'
    else:
        model_name = 'xlm-roberta-base'

    tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_name)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_name)

    def preprocess_function(examples):
        return tokenizer(examples['sentence'], padding=False, truncation=True)

    sst_train_dataset = sst_train_dataset.map(preprocess_function, batched=True)
    sst_validation_dataset = sst_validation_dataset.map(preprocess_function, batched=True)

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
        train_dataset=sst_train_dataset,
        eval_dataset=sst_validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, preds, average='binary')
    acc = metrics.accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == '__main__':
    parser = create_arg_parser()
    args = parser.parse_args()
    main(
        args.output_dir,
        args.logging_dir,
        args.logging_steps,
        args.large,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate,
        args.num_train_epochs,
        args.warmup_ratio)
