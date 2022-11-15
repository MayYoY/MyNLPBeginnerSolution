import numpy as np
import pandas as pd
import re
import d2l.torch as d2l
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm.auto import tqdm

from data import utils
from nn import classifier


class Config:
    num_epochs = 50
    lr = 5e-3
    batch_size = 96
    num_class = 5


class Accumulate:
    def __init__(self):
        self.cnt = 0
        self.val = 0
        self.sum = 0
        self.avg = 0

    def reset(self):
        self.cnt = 0
        self.val = 0
        self.sum = 0
        self.avg = 0

    def update(self, val, n=1):
        self.cnt += n
        self.val = val
        self.sum += val
        self.avg = self.sum / self.cnt


def train(model, data_iter, num_epochs):
    """
    训练softmax回归模型
    :param model:
    :param data_iter:
    :param num_epochs:
    :return:
    """
    num_training_steps = num_epochs * len(data_iter)
    progress_bar = tqdm(range(num_training_steps))
    metric = Accumulate()

    model.train = True
    for epoch in range(num_epochs):
        metric.reset()
        for X, y in data_iter:
            pred = model.forward(X, y)
            loss = classifier.loss_fun(pred, y)
            model.step()
            metric.update(loss)
            progress_bar.update(1)
        print(f'Epoch{epoch + 1}: avg_loss = {metric.avg}')


def evaluateIter(model, data_iter):
    model.train = False
    acc_metric = Accumulate()
    acc_metric.reset()
    loss_metric = Accumulate()
    loss_metric.reset()
    for x, y in data_iter:
        pred = model.forward(x)
        correct_count = classifier.correctCount(pred, y)
        loss = classifier.loss_fun(pred, y)
        acc_metric.update(correct_count, len(y))
        loss_metric.update(loss)
    print(f'Accuracy: {acc_metric.avg:.2f}\tAverage loss: {loss_metric.avg:.2f}')


# features, labels = utils.read_data()
data = pd.read_csv('train.tsv', sep='\t')  # 带标签, 方便评估
labels = data['Sentiment'].to_numpy()
corpus = data['Phrase'].tolist()  # len = 156060

# train_set = utils.Dataset(features, labels)
train_set = utils.TextDataset(corpus, labels)
train_iter = utils.DataLoader(train_set, shuffle=True, batch_size=Config.batch_size)

# model = classifier.Model(features.shape[1], Config.num_class, Config.lr)
model = classifier.Model(len(train_set.vocab.token_to_idx), Config.num_class, Config.lr)

print(f'Before training:')
# classifier.evaluate(model, features, labels)
evaluateIter(model, train_iter)
train(model, train_iter, Config.num_epochs)
print('************************')
print(f'After training')
# classifier.evaluate(model, features, labels)
evaluateIter(model, train_iter)


"""data = pd.read_csv('train.tsv', sep='\t')  # 带标签, 方便评估
labels = data['Sentiment'].to_numpy()
corpus = data['Phrase'].tolist()  # len = 156060

ds = utils.TextDataset(corpus, labels)
ds_iter = utils.DataLoader(ds, shuffle=True, batch_size=Config.batch_size)
for x, y in ds_iter:
    continue"""
