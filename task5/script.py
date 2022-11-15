"""
用LSTM、GRU来训练字符级的语言模型，计算困惑度

1. 参考
   1. 《[神经网络与深度学习](https://nndl.github.io/)》 第6、15章
2. 数据集：poetryFromTang.txt
3. 实现要求：Pytorch
4. 知识点：
   1. 语言模型：困惑度等
   2. 文本生成
5. 时间：两周
"""

import torch
import torch.nn as nn
from dataset import loader
from torch.utils import data
from model import languageModel, metric
from dataset import textHelp


sentence_path = 'poetrySentenceDataset.txt'
poem_path = 'poetryDataset.txt'
sentence_csv = "sentenceForText.csv"
poem_csv = "poetryForText.csv"
train_path = "poetryTrain.txt"
eval_path = "poetryEval.txt"
train_sentence_path = "sentenceTrain.txt"
eval_sentence_path = "sentenceEval.txt"


class Config:
    batch_size = 64
    num_epochs = 10  # 20 for poem
    lr = 0.005  # 0.005 for poem, still overfitting
    embed_size = 1024
    hidden_size = 1024
    device = "cuda"
    loss_fun = metric.MaskedSoftmaxCELoss()
    # loss_fun_text = nn.CrossEntropyLoss(reduction="none")
    loss_fun_text = nn.CrossEntropyLoss()


myset = loader.PoetryDataset(train_path, max_len=175)  # poem 175
myiter = data.DataLoader(myset, batch_size=Config.batch_size, shuffle=True)
eval_set = loader.PoetryDataset(eval_path, max_len=175)
eval_iter = data.DataLoader(eval_set, batch_size=Config.batch_size, shuffle=False)

net = languageModel.MyLM(len(myset.vocab), Config.embed_size, Config.hidden_size, rnn_type="GRU")
optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr, weight_decay=0.01)

languageModel.train(net, Config.loss_fun, optimizer, Config.num_epochs,
                    myiter, eval_iter, Config.device)
torch.save(net.state_dict(), "./LSTM_175_poem.pt")
# myset = loader.PoetryDataset(train_path, max_len=175)
# net = languageModel.MyLM(len(myset.vocab), Config.embed_size, Config.hidden_size)
# net.load_state_dict(torch.load("./LSTM_175_poem.pt"))

input_seq = ["<bos>", "落", "木"]
pred_seq = languageModel.generate(net, input_seq=input_seq, vocab=myset.vocab,
                                  device=Config.device)
print(pred_seq)

"""myset = loader.PoetryDataset(train_sentence_path, max_len=16)
myiter = data.DataLoader(myset, batch_size=Config.batch_size, shuffle=True)
eval_set = loader.PoetryDataset(eval_sentence_path, max_len=16)
eval_iter = data.DataLoader(eval_set, batch_size=Config.batch_size, shuffle=False)

net = languageModel.MyLM(len(myset.vocab), Config.embed_size, Config.hidden_size)
optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr)

languageModel.train(net, Config.loss_fun, optimizer, Config.num_epochs,
                    myiter, eval_iter, Config.device)
torch.save(net.state_dict(), "./LSTM_16_sentence.pt")
myset = loader.PoetryDataset(train_sentence_path, max_len=16)
net = languageModel.MyLM(len(myset.vocab), Config.embed_size, Config.hidden_size)
net.load_state_dict(torch.load("./LSTM_16_sentence.pt"))
pred_seq = languageModel.generate(net, input_seq=["<bos>", "忽", "如", "一"], vocab=myset.vocab,
                                  device=Config.device)
print(pred_seq)"""


"""train_iter, vocab = textHelp.getDataset(poem_csv, Config.batch_size)
net = languageModel.MyLM(len(vocab), Config.embed_size, Config.hidden_size, rnn_type="GRU")
net.apply(net.init_weight)
optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr)
languageModel.trainForText(net, Config.loss_fun_text, optimizer, Config.num_epochs,
                           train_iter, train_iter, Config.device)
pred_seq = languageModel.generate(net, input_seq="君不见", vocab=vocab,
                                  device=Config.device)
print(pred_seq)

# languageModel.train(net, Config.loss_fun, optimizer, Config.num_epochs, train_iter,
# train_iter, Config.device)
# 针对 torchtext 的训练和评估"""
