import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import jieba
import os
import torch.utils.data as D
import time
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def construct_list(filepath):
    text = open(filepath, "r",encoding="utf-8").readlines()
    list_of_tuples = []
    for sentence_and_label in text:
        temp = sentence_and_label.split("\t")
        temp_tuple = (temp[0], int(temp[1][0]))
        list_of_tuples.append(temp_tuple)
    return list_of_tuples

#print(construct_list("./corpus/test.txt"))
list_of_stop_word = open("./corpus/stop_words.txt", "r", encoding="utf-8").readlines()
list_of_stop_word = [line.strip("\n") for line in list_of_stop_word]
#print(list_of_stop_word)

def list_of_sentence_splitted(filepath):
    sentence_and_label_list = construct_list(filepath)
    list_of_sentence_splitted = []
    for sentence_and_label in sentence_and_label_list:
        sentence = sentence_and_label[0]
        sentence_splitted = jieba.lcut(sentence)
        temp_list = []
        for word in sentence_splitted:
            if word in list_of_stop_word:
                continue
            else:
                temp_list.append(word)
        list_of_sentence_splitted.append(temp_list)
    return list_of_sentence_splitted
#print(list_of_sentence_splitted("./corpus/test.txt"))

def construct_dict(list_of_sentences_splitted):
    wordlist = []
    for sentence in list_of_sentences_splitted:
        for word in sentence:
            wordlist.append(word)
    wordset = set(wordlist)
    wordlist = list(wordset) + ['<PAD>']
    worddict_word_to_index = {}
    worddict_index_to_word = {}
    for i in range(0, len(wordlist)):
        worddict_word_to_index[wordlist[i]] = i
        worddict_index_to_word[i] = wordlist[i]
    worddict_word_to_index['<PAD>'] = len(wordlist)
    worddict_index_to_word[len(wordlist)] = '<PAD>'
    return worddict_word_to_index, worddict_index_to_word
w2i, i2w = construct_dict(list_of_sentence_splitted("./corpus/train.txt")
                          +list_of_sentence_splitted("./corpus/test.txt")
                          +list_of_sentence_splitted("./corpus/dev.txt"))
#print(w2i)
#print(i2w)

def compute_feature_and_label(filepath):
    max_length = 0
    list_of_list = list_of_sentence_splitted(filepath)
    for sentence in list_of_list:
        length = len(sentence)
        if length > max_length:
            max_length = length
    for sentence in list_of_list:
        for i in range(0, max_length - len(sentence)):
            sentence.append('<PAD>')

    feature = []
    for sentence in list_of_list:
        index_list = []
        for word in sentence:
            index = w2i[word]
            index_list.append(index)
        feature.append(index_list)
    tensor_of_feature = torch.IntTensor(feature)
    label = [sentence[1] for sentence in construct_list(filepath)]
    tensor_of_label = torch.LongTensor(label)
    return tensor_of_feature, tensor_of_label
#feature, label = compute_feature_and_label("./corpus/train.txt")
#print(feature.shape)
#print(feature)
#print(label.shape)
#print(label)
train_feature, train_label = compute_feature_and_label("./corpus/train.txt")
test_feature, test_label = compute_feature_and_label("./corpus/test.txt")
dev_feature, dev_label = compute_feature_and_label("./corpus/dev.txt")
train_dataset = D.TensorDataset(train_feature, train_label)
test_dataset = D.TensorDataset(test_feature, test_label)
dev_dataset = D.TensorDataset(dev_feature, dev_label)
train_dataloader = D.DataLoader(train_dataset, batch_size=32)
test_dataloader = D.DataLoader(test_dataset, batch_size=32)
dev_dataloader = D.DataLoader(dev_dataset, batch_size=32)

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
         # x shape: (batch_size, channel, seq_len)
        return F.max_pool1d(x, kernel_size=x.shape[2]) # shape: (batch_size, channel, 1)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, kernel_sizes, num_channels):
        super(TextCNN, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size+1, embedding_dim, padding_idx=w2i['<PAD>'])
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(num_channels), 4)
        self.pool = GlobalMaxPool1d()
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(in_channels = embedding_dim, 
                                        out_channels = c, 
                                        kernel_size = k))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = embeds.permute(0, 2, 1)
        encoding = torch.cat([self.pool(F.relu(conv(embeds))).squeeze(-1) for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch, label in data_iter:
            X, y = batch, label
            net.eval() # 评估模式, 这会关闭dropout
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            net.train() # 改回训练模式 
            n += y.shape[0]
    return acc_sum / n

#for data_batch, target_batch in train_dataloader:
#    print(data_batch)
#    print(target_batch)

def train(train_iter, test_iter, net, loss, optimizer, num_epochs):
    batch_count = 0
    prev_acc = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for batch, label in tqdm(train_iter):
            X, y = batch, label
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               test_acc, time.time() - start))
        if(epoch % 5 == 0):
            acc = evaluate_accuracy(dev_dataloader, net)
            if(acc < prev_acc):
                break
            else:
                prev_acc = acc


lr, num_epochs = 0.001, 100
net = TextCNN(len(w2i), embedding_dim=200, kernel_sizes=[3, 4, 5], num_channels=[100, 100, 100])
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
train(train_dataloader, test_dataloader, net, loss, optimizer, num_epochs)
test_acc = evaluate_accuracy(test_dataloader, net)
print(test_acc)
