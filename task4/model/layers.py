import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import utils


class CharEmbedCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, kernel_size, filter_num):
        super(CharEmbedCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.filter_num = filter_num
        # output: B x out_channels x (cl - kernel_size + 1)
        self.conv = nn.Conv1d(kernel_size=kernel_size,  # 仅在字符维度作卷积
                              in_channels=embed_size,  # embed 维度作为通道维
                              out_channels=embed_size * filter_num)

        utils.init_embedding(self.embed)

    def forward(self, x):
        # x: B x T x char_len
        B, T, _ = x.shape
        x = self.embed(x).permute(0, 1, 3, 2)  # B x T x embed_size x cl
        y = torch.zeros(B, T, self.embed_size * self.filter_num, device=x.device)  # B x T x embed_size*filter_num
        for i in range(T):
            y[:, i, :], _ = torch.max(self.conv(x[:, i, :, :]), dim=2)
        return y


class CharEmbedBLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(CharEmbedBLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: B x T x char_len
        B, T, _ = x.shape
        x = self.embed(x).permute(0, 2, 1, 3)  # B x cl x T x embed_size
        y = torch.zeros(B, T, 2 * self.hidden_size, device=x.device)
        for i in range(B):
            x[i, :], _ = self.rnn(x[i, :])  # B x cl x T x 2*hidden_size
            y[i, :, :self.hidden_size] = x[i, -1, :, :self.hidden_size]  # 最后一个字符的前向输出
            y[i, :, self.hidden_size:] = x[i, 0, :, self.hidden_size:]  # 第一个字符的反向输出
        return y
