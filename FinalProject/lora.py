import logging
import os
import wandb
import sys
import peft
import evaluate
import argparse
from nltk.tokenize import sent_tokenize
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

parser.add_argument('-s', '--seed', type=int)
parser.add_argument('-lr', '--learning_rate', type=float)
parser.add_argument('-e', '--num_train_epochs', type=int)
parser.add_argument('-d', '--dataset_path', type=str)
parser.add_argument('-m', '--model_path', type=str)
parser.add_argument('-n', '--name', type=str)
parser.add_argument('-b', '--batch_size', type=int)
parser.add_argument('-o', '--output_dir', type=str)
arguments = parser.parse_args()

max_input_length = 500
max_target_length = 100

tokenizer = AutoTokenizer.from_pretrained(arguments.model_path, local_files_only=True)
rouge_score = evaluate.load("./metrics/rouge")

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["content"],
        max_length=max_input_length,
        truncation=True
    )
    labels = tokenizer(
        examples["title"],
        max_length=max_target_length,
        truncation=True
    )
    model_inputs["labels"]=labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )

    result = {key: value * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

def main():

    wandb.init(
        project="NLPDL-FinalProject",

        config={
            "learning_rate": arguments.learning_rate,
            "epochs": arguments.num_train_epochs
        },

        name="seed="+(str)(arguments.seed)
    )

    os.environ["WANDB_MODE"]="offline"
    os.environ["WANDB_LOG_MODEL"]="checkpoint"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set seed before initializing model.
    set_seed(arguments.seed)

    # load the tokenizer from the arguments
    raw_datasets = load_dataset("json", data_files=arguments.dataset_path, field="data")
    raw_datasets = raw_datasets["train"]
    raw_datasets = raw_datasets.train_test_split(train_size=0.9)
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

    config = AutoConfig.from_pretrained(arguments.model_path, local_files_only=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(arguments.model_path, config=config, local_files_only=True)

    logging_steps = len(tokenized_datasets["train"]) // arguments.batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"{arguments.name}",
        evaluation_strategy="epoch",
        learning_rate=arguments.learning_rate,
        per_device_eval_batch_size=arguments.batch_size,
        per_device_train_batch_size=arguments.batch_size,
        gradient_accumulation_steps=4,
        weight_decay=0.01,
        optim="adafactor",
        report_to="wandb",
        num_train_epochs=arguments.num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=False
    )

    peft_config = peft.LoraConfig(
        task_type=peft.utils.peft_types.TaskType.SEQ_2_SEQ_LM,
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )
    
    model = peft.get_peft_model(model, peft_config)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    trainer.train()

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    wandb.finish()

if __name__ == "__main__":
    main()
