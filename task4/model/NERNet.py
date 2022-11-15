import torch
import torch.nn as nn
import torch.nn.functional as F
from TorchCRF import CRF
from tqdm.auto import tqdm

from . import layers, conlleval
from dataset import utils
from dataset import loader


def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项
    :param X: B x T
    :param valid_len: B x 1
    :param value:
    :return mask: B x T
    """
    max_len = X.size(1)  # T
    # None, : 和 :, None 是维度扩充, 将 tensor 变为 1 x T 和 B x 1
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # X[~mask] = value
    return mask


class BLSTM_CNN_CRF(nn.Module):
    def __init__(self, label_num: int, token_num: int, char_num: int,
                 word_embed_size: int, char_embed_size: int, hidden_size: int,
                 char_kernel_size: int, char_filter_num: int,
                 pretrained_path=None, token_vocab=None):
        super(BLSTM_CNN_CRF, self).__init__()
        self.word_embed = nn.Embedding(token_num, word_embed_size)
        self.char_embed = layers.CharEmbedCNN(char_num, char_embed_size,
                                              char_kernel_size, char_filter_num)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = nn.LSTM(input_size=word_embed_size + char_embed_size * char_filter_num,
                           hidden_size=hidden_size, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_size, label_num)
        self.crf = CRF(label_num)
        self.hidden_size = hidden_size

        # 初始化权重
        utils.init_lstm(self.rnn)
        utils.init_embedding(self.word_embed, pretrained_path=pretrained_path,
                             vocab=token_vocab)
        utils.init_linear(self.fc)

    def log_likelihood(self, x, x_len, x_char, y):
        B, T = x.shape
        mask = sequence_mask(x, x_len)
        x = self.word_embed(x)  # B x T x we
        x = self.dropout(x)
        x_char = self.char_embed(x_char)
        x_char = self.dropout(x_char)
        x = torch.cat((x, x_char), dim=2).permute(1, 0, 2)  # T x B x (we + ce*filter_num)
        x, _ = self.rnn(x)
        x = x.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size)  # B*T x 2*hidden_size
        pred = self.fc(x).reshape(B, T, -1)  # B x T x label_num
        loss = -1 * self.crf(pred, y, mask)
        return loss

    def forward(self, x, x_len, x_char):
        B, T = x.shape
        mask = sequence_mask(x, x_len)
        x = self.word_embed(x)  # B x T x we
        x = self.dropout(x)
        x_char = self.char_embed(x_char)
        x_char = self.dropout(x_char)
        x = torch.cat((x, x_char), dim=2).permute(1, 0, 2)  # T x B x (we + ce*filter_num)
        x, _ = self.rnn(x)
        x = x.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size)  # B*T x 2*hidden_size
        pred = self.fc(x).reshape(B, T, -1)  # B x T x label_num
        pred = self.crf.viterbi_decode(pred, mask)
        return pred


def train(net: BLSTM_CNN_CRF, optimizer: torch.optim.Optimizer, lr,
          num_epochs: int, train_iter, eval_iter, label_vocab: loader.Vocab, device):
    progress_bar = tqdm(range(num_epochs * len(train_iter)))
    acc = utils.Accumulator()
    net.to(device)
    for i in range(num_epochs):
        net.train()  # 调回 train
        for x, x_len, x_char, y in train_iter:
            optimizer.zero_grad()  # 归零
            x = x.to(device)
            x_len = x_len.to(device)
            x_char = x_char.to(device)
            y = y.to(device)

            loss = net.log_likelihood(x, x_len, x_char, y)
            loss.mean().backward()  # mean or sum
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=5)  # 梯度截断
            optimizer.step()  # 更新
            # 更新进度
            acc.update(len(x), loss.sum())
            progress_bar.update(1)
        # 更新学习率
        utils.adjust_learning_rate(optimizer=optimizer, lr=lr / (1 + 0.05 * (i + 1)))
        print(f'Epoch {i + 1}: average loss: {acc.getAvg():.3f}')
    evaluation(net, eval_iter, label_vocab, device)


def evaluation(net: BLSTM_CNN_CRF, eval_iter, label_vocab: loader.Vocab, device):
    net.to(device)
    net.eval()
    true_seqs = []
    pred_seqs = []
    for x, x_len, x_char, y in eval_iter:
        x = x.to(device)
        x_len = x_len.to(device)
        x_char = x_char.to(device)
        y = y.to(device)
        preds = net(x, x_len, x_char)  # list(list(int)), B x valid_len
        for i, pred in enumerate(preds):
            for j, word_pred in enumerate(pred):
                pred_seqs.append(label_vocab.idx_to_token[word_pred])
                true_seqs.append(label_vocab.idx_to_token[y[i, j]])
            pred_seqs.append('O')  # 用于分隔两个句子
            true_seqs.append('O')

    result = conlleval.evaluate(true_seqs, pred_seqs)
    print('result:    ', result)
