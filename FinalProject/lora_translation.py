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

max_length = 512
source_lang = "zh"
target_lang = "en"
prefix = "translate Chinese to English: "

tokenizer = AutoTokenizer.from_pretrained(arguments.model_path, local_files_only=True)
bleu_score = evaluate.load("/ceph/home/wangkeran/NLPProj/metrics/sacrebleu/sacrebleu.py")

def preprocess_function(examples):
    inputs = [prefix + example for example in examples[source_lang]]
    targets = [example for example in examples[target_lang]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
    return model_inputs

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    print(decoded_preds)
    decoded_labels = [[label.strip()] for label in decoded_labels]
    print(decoded_labels)

    result = bleu_score.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

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
    raw_datasets = load_dataset("json", data_files=arguments.dataset_path)
    raw_datasets = raw_datasets['train']
    raw_datasets = raw_datasets.train_test_split(train_size=0.9)

    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, remove_columns=raw_datasets["train"].column_names)

    config = AutoConfig.from_pretrained(arguments.model_path, local_files_only=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(arguments.model_path, config=config, local_files_only=True)

    logging_steps = len(tokenized_datasets["train"]) // arguments.batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"{arguments.name}",
        evaluation_strategy="no",
        save_strategy="epoch",
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
