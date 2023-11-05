import numpy as np
import torch
import scipy
import math
import time
import sys
from collections import Counter

def getCorpus(filetype, size):
    if filetype == 'dev':
        filepath = '../corpus/text8.dev.txt'
    elif filetype == 'test':
        filepath = '../corpus/text8.test.txt'
    else:
        filepath = '../corpus/text8.train.txt'

    with open(filepath, "r") as f:
        text = f.read()
        text = text.lower().split()
        text = text[: min(len(text), size)]
        vocab_dict = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
        vocab_dict['<unk>'] = len(text) - sum(list(vocab_dict.values()))
        idx_to_word = list(vocab_dict.keys())
        word_to_idx = {word:ind for ind, word in enumerate(idx_to_word)}
        word_counts = np.array(list(vocab_dict.values()), dtype=np.float32)
        word_freqs = word_counts / sum(word_counts)
        print("Words list length:{}".format(len(text)))
        print("Vocab size:{}".format(len(idx_to_word)))
    return text, idx_to_word, word_to_idx, word_counts, word_freqs

def buildCooccuranceMatrix(text, word_to_idx):
    vocab_size = len(word_to_idx)
    maxlength = len(text)
    text_ids = [word_to_idx.get(word, word_to_idx["<unk>"]) for word in text]
    cooccurance_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    print("Co-Matrix consumed mem:%.2fMB" % (sys.getsizeof(cooccurance_matrix)/(1024*1024)))
    for i, center_word_id in enumerate(text_ids):
        window_indices = list(range(i - WINDOW_SIZE, i)) + list(range(i + 1, i + WINDOW_SIZE + 1))
        window_indices = [i % maxlength for i in window_indices]
        window_word_ids = [text_ids[index] for index in window_indices]
        for context_word_id in window_word_ids:
            cooccurance_matrix[center_word_id][context_word_id] += 1
        if (i+1) % 1000000 == 0:
            print(">>>>> Process %dth word" % (i+1))
    print(">>>>> Save co-occurance matrix completed.")
    return cooccurance_matrix

def buildWeightMatrix(co_matrix):
    xmax = 100.0
    weight_matrix = np.zeros_like(co_matrix, dtype=np.float32)
    print("Weight-Matrix consumed mem:%.2fMB" % (sys.getsizeof(weight_matrix) / (1024 * 1024)))
    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            weight_matrix[i][j] = math.pow(co_matrix[i][j] / xmax, 0.75) if co_matrix[i][j] < xmax else 1
        if (i+1) % 1000 == 0:
            print(">>>>> Process %dth weight" % (i+1))
    print(">>>>> Save weight matrix completed.")
    return weight_matrix

class WordEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, co_matrix, weight_matrix):
        self.co_matrix = co_matrix
        self.weight_matrix = weight_matrix
        self.train_set = []

        for i in range(self.weight_matrix.shape[0]):
            for j in range(self.weight_matrix.shape[1]):
                if weight_matrix[i][j] != 0:
                    # 这里对权重进行了筛选，去掉权重为0的项 
                    # 因为共现次数为0会导致log(X)变成nan
                    self.train_set.append((i, j))   

    def __len__(self):
        '''
        必须重写的方法
        :return: 返回训练集的大小
        '''
        return len(self.train_set)

    def __getitem__(self, index):
        '''
        必须重写的方法
        :param index:样本索引 
        :return: 返回一个样本
        '''
        (i, j) = self.train_set[index]
        return i, j, torch.tensor(self.co_matrix[i][j], dtype=torch.float), self.weight_matrix[i][j]

class GloveModelForBGD(torch.nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        #声明v和w为Embedding向量
        self.v = torch.nn.Embedding(vocab_size, embed_size)
        self.w = torch.nn.Embedding(vocab_size, embed_size)
        self.biasv = torch.nn.Embedding(vocab_size, 1)
        self.biasw = torch.nn.Embedding(vocab_size, 1)
        
        #随机初始化参数
        initrange = 0.5 / self.embed_size
        self.v.weight.data.uniform_(-initrange, initrange)
        self.w.weight.data.uniform_(-initrange, initrange)

    def forward(self, i, j, co_occur, weight):
    	#根据目标函数计算Loss值
        vi = self.v(i)	#分别根据索引i和j取出对应的词向量和偏差值
        wj = self.w(j)
        bi = self.biasv(i)
        bj = self.biasw(j)

        similarity = torch.mul(vi, wj)
        similarity = torch.sum(similarity, dim=1)

        loss = similarity + bi + bj - torch.log(co_occur)
        loss = 0.5 * weight * loss * loss

        return loss.sum().mean()

    def gloveMatrix(self):
        '''
        获得词向量，这里把两个向量相加作为最后的词向量
        :return: 
        '''
        return self.v.weight.data.numpy() + self.w.weight.data.numpy()

EMBEDDING_SIZE = 50		#50个特征
MAX_VOCAB_SIZE = 2000	#词汇表大小为2000个词语
WINDOW_SIZE = 5			#窗口大小为5

NUM_EPOCHS = 10			#迭代10次
BATCH_SIZE = 10			#一批有10个样本
LEARNING_RATE = 0.05	#初始学习率
TEXT_SIZE = 20000000	#控制从语料库读取语料的规模
WEIGHT_FILE = "weight.txt"

text, idx_to_word, word_to_idx, word_counts, word_freqs = getCorpus('train', size=TEXT_SIZE)    #加载语料及预处理
co_matrix = buildCooccuranceMatrix(text, word_to_idx)    #构建共现矩阵
weight_matrix = buildWeightMatrix(co_matrix)             #构建权重矩阵
dataset = WordEmbeddingDataset(co_matrix, weight_matrix) #创建dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
model = GloveModelForBGD(MAX_VOCAB_SIZE, EMBEDDING_SIZE) #创建模型
optimizer = torch.optim.Adagrad(model.parameters(), lr=LEARNING_RATE) #选择Adagrad优化器

print_every = 10000
save_every = 50000
epochs = NUM_EPOCHS
iters_per_epoch = int(dataset.__len__() / BATCH_SIZE)
total_iterations = iters_per_epoch * epochs
print("Iterations: %d per one epoch, Total iterations: %d " % (iters_per_epoch, total_iterations))
start = time.time()
for epoch in range(epochs):
    loss_print_avg = 0
    iteration = iters_per_epoch * epoch
    for i, j, co_occur, weight in dataloader:
        iteration += 1
        optimizer.zero_grad()   #每一批样本训练前重置缓存的梯度
        loss = model(i, j, co_occur, weight)    #前向传播
        loss.backward()     #反向传播
        optimizer.step()    #更新梯度
        loss_print_avg += loss.item()
torch.save(model.state_dict(), WEIGHT_FILE)
