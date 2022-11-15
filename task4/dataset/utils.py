import torch
import numpy as np
import torch.nn as nn


class Accumulator:
    def __init__(self):
        self.count = 0
        self.total_loss = 0.

    def reset(self):
        self.count = 0
        self.total_loss = 0.

    def update(self, n, loss):
        self.count += n
        self.total_loss += loss

    def getAvg(self):
        return self.total_loss / self.count


def iob(label):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    IOB1: 标签 B 仅用于两个连续的同类型命名实体的边界区分，不用于命名实体的起始位置
    """
    for i, word_label in enumerate(label):
        if word_label == 'O':
            continue
        split = word_label.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False, label
        if split[0] == 'B':  # 属于 IOB2
            continue
        elif i == 0 or label[i - 1] == 'O':  # conversion IOB1 to IOB2
            # i == 0 但 tag 不是 B; i != 0 且前一个词不属于这个命名实体
            label[i] = 'B' + word_label[1:]
        elif label[i - 1][1:] == word_label[1:]:  # 前一个词也属于这个命名实体
            continue
        else:  # conversion IOB1 to IOB2
            label[i] = 'B' + word_label[1:]
    return True, label


def iob2iobes(label):
    """
    IOB -> IOBES
    """
    new_label = []
    for i, word_label in enumerate(label):
        if word_label == 'O':
            new_label.append(word_label)
        elif word_label.split('-')[0] == 'B':
            if i + 1 != len(label) and label[i + 1].split('-')[0] == 'I':
                new_label.append(word_label)
            else:  # singleton
                new_label.append(word_label.replace('B-', 'S-'))
        elif word_label.split('-')[0] == 'I':
            if i + 1 < len(label) and label[i + 1].split('-')[0] == 'I':
                new_label.append(word_label)
            else:  # end
                new_label.append(word_label.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_label


def iobes2iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, pretrained_path):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(pretrained_path)
        self.unknown_idx = 0
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, pretrained_path):
        idx_to_token, idx_to_vec = ['<unk>'], []
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(pretrained_path, encoding='utf-8', mode='r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [
            self.token_to_idx.get(token, self.unknown_idx)
            for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)


def init_embedding(input_embedding: nn.Embedding, pretrained_path=None, vocab=None):
    """
    Initialize embedding
    """
    if pretrained_path is None or vocab is None:
        bias = np.sqrt(3.0 / input_embedding.weight.size(1))
        nn.init.uniform_(input_embedding.weight, -bias, bias)
    else:
        pretrained_embed = TokenEmbedding(pretrained_path)
        embeds = pretrained_embed[vocab.idx_to_token]
        input_embedding.weight.data.copy_(embeds)


def init_linear(input_linear: nn.Linear):
    """
    Initialize linear transformation
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm: nn.LSTM):
    """
    Initialize lstm
    """
    for idx in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(idx))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(idx))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -bias, bias)
    if input_lstm.bidirectional:
        for idx in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(idx) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(idx) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -bias, bias)

    if input_lstm.bias:
        for idx in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(idx))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(idx))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for idx in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(idx) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(idx) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
