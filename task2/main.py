import torch
import torch.nn as nn
from torchtext.legacy import data
import numpy as np
import random
import os

from model import classifier

import d2l.torch as d2l

"""
word embedding
1. 随机 embedding 的初始化方式
2. 用 glove 预训练的 embedding 进行初始化 https://nlp.stanford.edu/projects/glove/
"""


class Config:
    seed = 42
    embed_size = 50  # 50, 100, 200, 300
    num_hidden = 100
    num_epochs = 30
    lr = 1e-3
    batch_size = 64
    batch_size_test = 32
    num_class = 5
    freeze = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


random.seed(Config.seed)
os.environ['PYTHONHASHSEED'] = str(Config.seed)
np.random.seed(Config.seed)
torch.manual_seed(Config.seed)
torch.cuda.manual_seed(Config.seed)
torch.cuda.manual_seed_all(Config.seed)
torch.backends.cudnn.deterministic = True


# Field
text_field = data.Field(sequential=True, batch_first=True, lower=True, pad_token="<pad>")
label_field = data.Field(sequential=False, batch_first=True, unk_token=None, use_vocab=False)

# Dataset
datafields = [("PhraseId", None), ("SentenceId", None),  # 不需要的filed设置为None
              ('Phrase', text_field), ('Sentiment', label_field)]
train_data = data.TabularDataset(path='train.csv', format='csv', fields=datafields,
                                 skip_header=True)
test_data = data.TabularDataset(path='test.csv', format='csv', fields=datafields,
                                skip_header=True)

# Vocab
text_field.build_vocab(train_data, vectors="glove.6B.50d",
                       unk_init=lambda x: torch.nn.init.uniform_(x, a=-0.25, b=0.25),
                       vectors_cache="glove.6B")
label_field.build_vocab(test_data)
pad_idx = text_field.vocab.stoi["<pad>"]
text_field.vocab.vectors[pad_idx] = 0.

# Iterator
train_iter = data.BucketIterator(train_data, batch_size=Config.batch_size,
                                 train=True, shuffle=True)  # , device=Config.device
test_iter = data.BucketIterator(test_data, batch_size=Config.batch_size_test, train=False,
                                sort=False)


"""model = classifier.MyCNN(len(text_field.vocab), Config.embed_size, Config.num_class,
                         method="glove", vectors=text_field.vocab.vectors,
                         freeze=Config.freeze, pad_idx=pad_idx)"""
model = classifier.MyRNN(len(text_field.vocab),
                         Config.embed_size, Config.num_class,
                         Config.num_hidden, method="random",
                         vectors=text_field.vocab.vectors,
                         freeze=Config.freeze, pad_idx=pad_idx)
# torch.load(model.state_dict(), "./result/rnn_glove_epoch30")

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=Config.lr)
loss_fun = nn.CrossEntropyLoss()

print(f'Before training:')
classifier.evaluate(model, test_iter, loss_fun, Config.device, Config.num_class)
print(f'Training...')
classifier.train(model, train_iter, test_iter, Config.num_epochs,
                 optimizer, loss_fun, Config.device, Config.num_class)
print(f'After training:')
classifier.evaluate(model, test_iter, loss_fun, Config.device, Config.num_class)
torch.save(model.state_dict(), "./result/rnn_random_epoch30_nostatic")

"""
考虑让 embedding 层 lr 更低
params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
trainer = torch.optim.SGD([{'params': params_1x},
                           {'params': net.fc.parameters(),
                            'lr': learning_rate * 10}],
                            lr=learning_rate, weight_decay=0.001)
"""
