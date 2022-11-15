import torch
import torch.nn as nn
from tqdm.auto import tqdm
import d2l.torch as d2l

from tools import SNLIDataset, preprocess
from model import esim, utils


train_path = './snli_1.0/snli_1.0_train.txt'
eval_path = './snli_1.0/snli_1.0_dev.txt'
test_path = './snli_1.0/snli_1.0_test.txt'


class Config:
    embed_size = 300  # 50, 100, 200, 300
    num_hidden = 300
    num_class = 3

    num_epochs = 50
    lr = 4e-4
    batch_size = 32
    freeze = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_text = SNLIDataset.read_snli(train_path)
train_data = SNLIDataset.SNLIDataSet(train_text)
train_iter = torch.utils.data.DataLoader(train_data, Config.batch_size, shuffle=True)

eval_text = SNLIDataset.read_snli(eval_path)
eval_set = SNLIDataset.SNLIDataSet(eval_text)
eval_iter = torch.utils.data.DataLoader(eval_set, Config.batch_size, shuffle=False)

net = esim.ESIM(len(train_data.vocab), embed_size=Config.embed_size, hidden_size=Config.num_hidden)
# 加载预训练embedding
glove_embedding = preprocess.TokenEmbedding('./glove/glove.6B.300d.txt')  # Config.embed_size
embeds = glove_embedding[train_data.vocab.idx_to_token]

optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
net.to(Config.device)
loss_fun = nn.CrossEntropyLoss()

net.load_state_dict(torch.load("./result/esim_checkpoint50"))  # ./result/esim_checkpoint20
# esim.train(net, train_iter, Config.num_epochs, optimizer, loss_fun, Config.device, train_iter)
esim.evaluate(net, eval_iter, loss_fun, Config.device)
