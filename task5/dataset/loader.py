import os
import torch
import torchtext
from torch.utils import data
import collections
from sklearn.model_selection import train_test_split


class Vocab:  # @save
    """文本词表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens):  # @save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


def preprocessTxt(read_path, write_path):
    file = open(read_path, encoding="utf-8")
    poem = ""
    target = open(write_path, encoding="utf-8", mode="w+")
    for line in file:
        if line == "\n":
            target.write(poem + "\n")
            poem = ""
        else:
            poem += line.rstrip('\n')


def readText(path):
    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]
    tokens = [[word for word in line] for line in lines]
    return tokens


def splitTrainEval(path, train_target, eval_target, test_size=0.2):
    lines = open(path, encoding="utf-8").readlines()
    lines = [line.rstrip("\n") for line in lines]
    X_train, X_eval = train_test_split(lines, test_size=test_size)
    train_lines = open(train_target, encoding="utf-8", mode="w+")
    eval_lines = open(eval_target, encoding="utf-8", mode="w+")
    for line in lines:
        train_lines.write(line + "\n")
    for line in lines:
        eval_lines.write(line + "\n")


def prepareSeq(tokens, vocab, max_len=100):
    """
    pad or truncate the sequences
    :param tokens:
    :param vocab:
    :param max_len: 30 or 16 for sentences, 125 or 230 or 350 for poems
    :return:
    """
    sequences = []
    valid_len = []
    for token in tokens:
        seq = [vocab["<bos>"]]
        if len(token) < max_len:
            for word in token:
                seq.append(vocab[word] if word in vocab.token_to_idx.keys()
                           else vocab["<unk>"])
            seq.append(vocab["<eos>"])
            while len(seq) < max_len + 2:
                seq.append(vocab["<pad>"])
        else:
            for i in range(max_len):
                seq.append(vocab[token[i]] if token[i] in vocab.token_to_idx.keys()
                           else vocab["<unk>"])
            seq.append(vocab["<eos>"])
        sequences.append(seq)
        valid_len.append(min(max_len, len(token)) + 2)  # 含 bos, eos

    return sequences, valid_len


class PoetryDataset(data.Dataset):
    def __init__(self, path='./poetryDataset.txt', min_freq=0, reserved_tokens=None, max_len=350):
        if reserved_tokens is None:
            reserved_tokens = ["<pad>", "<bos>", "<eos>"]
        super(PoetryDataset, self).__init__()
        tokens = readText(path)
        self.vocab = Vocab(tokens, min_freq=min_freq, reserved_tokens=reserved_tokens)
        self.sequences, self.valid_len = prepareSeq(tokens, self.vocab, max_len)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.sequences[idx])
        x_len = torch.tensor(self.valid_len[idx])
        return x, x_len

    def __len__(self):
        return len(self.sequences)
