import torch
import torch.nn as nn
import math
import collections


class Accumulator:
    def __init__(self):
        """ self.last_loss = 0
         self.last_bleu = 0
        self.total_bleu = 0."""
        self.count = 0
        self.total_loss = 0.

    def reset(self):
        """self.last_loss = 0
        self.last_bleu = 0
        self.total_bleu = 0."""
        self.count = 0
        self.total_loss = 0.

    def update(self, n, loss):
        self.count += n
        # self.total_bleu += bleu
        self.total_loss += loss

    def getAvg(self):
        return self.total_loss / self.count

    def getPPL(self):
        # perplexity = \exp(-\frac{1}{n}\sum_{i=1}^n\log P(x_i|x_{i-1}, ..., x_1))
        # 也就是对平均交叉熵作对数, 起到改变量纲的作用
        return math.exp(self.total_loss / self.count)


def sequence_mask(X, valid_len, value=0):
    """
    在序列中屏蔽不相关的项
    :param X: B x T x vocab_size
    :param valid_len: B x 1
    :param value:
    :return:
    """
    max_len = X.size(1)  # T
    # None, : 和 :, None 是维度扩充, 将 tensor 变为 1 x T 和 B x 1
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带掩码的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'  # 不求平均或求和, 直接返回损失 tensor
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
                                pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)  # B x T -> B
        # weighted_loss = unweighted_loss * weights  # B x T
        return weighted_loss


def bleu(pred_seq, label_seq, k):
    """
    计算困惑度
    BLEU = \exp(\min(0, 1 - \frac{len_{label}}{len_{pred}})) \prod_{n=1}^{k}p_{n}^{2^{-n}}
    k 为用于匹配序列的最长 n 元语法
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))  # 第一项 系数
    # 第二项 连乘
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1  # 统计 label 中有的 n 元语法
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:  # 匹配预测序列中的 n 元语法
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
