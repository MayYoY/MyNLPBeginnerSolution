import torch
import matplotlib.pyplot as plt
from torch.utils import data
from dataset import loader, utils
from model import NERNet

import d2l.torch as d2l


train_path = "./CoNNL_2003/train.txt"
eval_path = "./CoNNL_2003/dev.txt"
test_path = "./CoNNL_2003/test.txt"


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


class Config:
    pretrained_path = "./glove/glove.6B.100d.txt"
    word_embed_size = 100
    char_embed_size = 30
    char_filter_num = 30
    char_kernel_size = 3
    hidden_size = 200

    lr = 0.015
    num_epochs = 50
    batch_size = 10
    momentum = 0.9
    device = try_gpu()


train_set = loader.NERDataset(train_path)
eval_set = loader.NERDataset(eval_path,
                             token_vocab=train_set.token_vocab,
                             label_vocab=train_set.label_vocab,
                             char_vocab=train_set.char_vocab)
train_iter = data.DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
eval_iter = data.DataLoader(eval_set, batch_size=Config.batch_size, shuffle=False)

label_num = len(train_set.label_vocab)
token_num = len(train_set.token_vocab)
char_num = len(train_set.char_vocab)
net = NERNet.BLSTM_CNN_CRF(label_num, token_num, char_num,
                           Config.word_embed_size,
                           Config.char_embed_size,
                           Config.hidden_size,
                           Config.char_kernel_size,
                           Config.char_filter_num,
                           pretrained_path=Config.pretrained_path,
                           token_vocab=train_set.token_vocab)

"""optimizer = torch.optim.SGD(net.parameters(), lr=Config.lr, momentum=Config.momentum)

NERNet.train(net, optimizer, Config.lr, Config.num_epochs,
             train_iter, eval_iter=eval_iter, label_vocab=train_set.label_vocab,
             device=Config.device)"""

net.load_state_dict(torch.load("./result/blstm_cnn_crf_checkpoint50"))
NERNet.evaluation(net, eval_iter, label_vocab=train_set.label_vocab, device=Config.device)
