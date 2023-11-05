from datasets import load_dataset
import sys
dataset = load_dataset("wikipedia","20220301.simple")
corpus = dataset['train']['text']
file_path = 'corpus.txt'
sys.stdout = open(file_path, "w")
print(corpus)