# NLPDL

Record the learning of NLPDL in PKU (2023 fall)

## Assignment1
Task2 Corpus:

```
from datasets import load_dataset
dataset = load_dataset("wikipedia", "20220301.simple")
corpus = dataset['train']['text']
```

## Assignment2
Task1 nmt github repo: https://github.com/linhaowei1/NLPDL/tree/main/Assignment_2/nmt

## Assignment3
the corpus and model are downloaded from huggingface and google drive:

[SemEval14-laptop/res_sup](https://drive.google.com/drive/folders/1H5rmibrg4VfEvM6uqobkrZlGla3xk78-?usp=share_link)

[acl_sup](https://github.com/UIC-Liu-Lab/ContinualLM)

[agnews_sup](https://huggingface.co/datasets/SetFit/ag_news)

[bert-base-uncased](https://huggingface.co/bert-base-uncased)

[roberta-base](https://huggingface.co/roberta-base)

[allenai/scibert_scivocab_uncased](https://huggingface.co/allenai/scibert_scivocab_uncased)

because of some network issues, I downloaded them and thus my code is internet-free

What's more, the graphs of loss/acc etc. can be check at the W&B site: [NLPDL-Assignment](https://wandb.ai/corkri/NLPDL-Assignment3?workspace=user-corkri)
