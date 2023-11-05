# All Import Statements Defined Here
# Note: Do not add to this list.
# ----------------

import sys
import nltk
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import pprint
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]
import numpy as np
import random
import scipy as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from ast import literal_eval

START_TOKEN = '<START>'
END_TOKEN = '<END>'

np.random.seed(0)
random.seed(0)
# ----------------

def read_corpus():
    """ Read files from the specified Reuter's category.
        Params:
            category (string): category name
        Return:
            list of lists, with words from each of the processed files
    """
    corpus = []
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    with open("D:/School_files/NLP/Assignment1/Task2/corpus.txt") as f:
        files = f.read()
        data = literal_eval(files)
    for para in data:
        sentences = para.split('\n\n')
        for sentence in sentences:
            corpus.append(sentence)
    return [[START_TOKEN] + [w.lower() for w in tokenizer.tokenize(f)] + [END_TOKEN] for f in corpus]

corpus = read_corpus()
num_of_ice = 0
num_of_steam = 0
list_for_ice = [0, 0, 0, 0]
list_for_steam = [0, 0, 0, 0]
window_size = 5

for sentence in corpus:
    index = 0
    length = len(sentence)
    for word in sentence:
        if word == 'ice':
            num_of_ice = num_of_ice + 1
            for i in range(max(index - window_size,0),index):
                if(sentence[i] == 'solid'):
                    list_for_ice[0] = list_for_ice[0] + 1
                elif(sentence[i] == 'gas'):
                    list_for_ice[1] = list_for_ice[1] + 1
                elif(sentence[i] == 'water'):
                    list_for_ice[2] = list_for_ice[2] + 1
                elif(sentence[i] == 'fashion'):
                    list_for_ice[3] = list_for_ice[3] + 1
            for i in range(index+1,min(index+window_size+1,length)):
                if(sentence[i] == 'solid'):
                    list_for_ice[0] = list_for_ice[0] + 1
                elif(sentence[i] == 'gas'):
                    list_for_ice[1] = list_for_ice[1] + 1
                elif(sentence[i] == 'water'):
                    list_for_ice[2] = list_for_ice[2] + 1
                elif(sentence[i] == 'fashion'):
                    list_for_ice[3] = list_for_ice[3] + 1
        if word == 'steam':
            num_of_steam = num_of_steam + 1
            for i in range(max(index - window_size,0),index):
                if(sentence[i] == 'solid'):
                    list_for_steam[0] = list_for_steam[0] + 1
                elif(sentence[i] == 'gas'):
                    list_for_steam[1] = list_for_steam[1] + 1
                elif(sentence[i] == 'water'):
                    list_for_steam[2] = list_for_steam[2] + 1
                elif(sentence[i] == 'fashion'):
                    list_for_steam[3] = list_for_steam[3] + 1
            for i in range(index+1,min(index+window_size+1,length)):
                if(sentence[i] == 'solid'):
                    list_for_steam[0] = list_for_ice[0] + 1
                elif(sentence[i] == 'gas'):
                    list_for_steam[1] = list_for_ice[1] + 1
                elif(sentence[i] == 'water'):
                    list_for_steam[2] = list_for_steam[2] + 1
                elif(sentence[i] == 'fashion'):
                    list_for_steam[3] = list_for_steam[3] + 1
        index = index + 1

print("num_of_ice = ",num_of_ice)
print("num_of_steam = ",num_of_steam)
print("list_for_ice = ",list_for_ice)
print("list_for_steam = ",list_for_steam)