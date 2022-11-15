import collections
import torch
from torch.utils import data
from . import utils


class Vocab:
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


def loadData(path):
    """
    加载数据集
    :return label: 标签
    :return tokens: 句子
    """
    lines = open(path, encoding='utf-8', mode='r+').readlines()[2:]
    lines = [line.rstrip('\n').split() for line in lines]
    tokens = []  # list(list(word))
    labels = []  # list(list(word_label)
    sentence = []
    label = []
    for line in lines:
        if not line:
            if sentence and label:
                tokens.append(sentence)
                labels.append(label)
            sentence = []
            label = []
        else:
            sentence.append(line[0])
            label.append(line[-1])
    chars = [[char for word in token for char in word] for token in tokens]
    return tokens, labels, chars


def updateLabelScheme(labels, scheme):
    """
    Only IOB1 and IOB2 schemes are accepted. 可将其转化为指定类型
    :param labels:
    :param scheme:
    :return:
    """
    for i, label in enumerate(labels):
        valid, label = utils.iob(label)
        if not valid:
            raise Exception('Sentences should be given in IOB format!')
        if scheme == "iob":  # update to iob2
            labels[i] = label
        elif scheme == "iobes":
            label = utils.iob2iobes(label)
            labels[i] = label
        else:
            raise Exception('Unsupported scheme!')
    return labels


def prepareSeq(tokens, labels, label_vocab, max_len=65):
    """
    pad or truncate the sequences
    :param tokens:
    :param labels:
    :param label_vocab:
    :param max_len:
    :return:
    """
    sequences = []
    valid_len = []
    labels_pad = []
    for i, token in enumerate(tokens):
        seq = []
        lab = []
        if len(token) < max_len:
            for j, word in enumerate(token):
                seq.append(word)
                lab.append(label_vocab[labels[i][j]])
            while len(seq) < max_len:
                seq.append("<pad>")
                lab.append(label_vocab["<pad>"])
        else:
            for j in range(max_len):
                seq.append(token[j])
                lab.append(label_vocab[labels[i][j]])
        sequences.append(seq)
        labels_pad.append(lab)
        valid_len.append(min(max_len, len(token)))

    return sequences, valid_len, labels_pad


def getTokenTensor(sequence, token_vocab):
    word_num = len(sequence)
    token_tensor = torch.zeros(word_num, dtype=torch.long)
    for i, word in enumerate(sequence):
        token_tensor[i] = token_vocab[word]
    return token_tensor


def getCharTensor(sequence, char_vocab, max_len=30):
    word_num = len(sequence)
    char_tensor = torch.zeros(word_num, max_len, dtype=torch.long)
    for i, word in enumerate(sequence):
        if word == "<pad>":
            continue
        else:
            for j, c in enumerate(word):
                if j >= max_len:
                    break
                char_tensor[i, j] = char_vocab[c]
    return char_tensor


class NERDataset(data.Dataset):
    def __init__(self, path, min_freq=0, reserved_tokens=None, max_len=65, char_len=30,
                 token_vocab=None, label_vocab=None, char_vocab=None):
        if reserved_tokens is None:
            reserved_tokens = ["<pad>"]
        super(NERDataset, self).__init__()
        tokens, labels, chars = loadData(path)
        labels = updateLabelScheme(labels, "iobes")

        if token_vocab is None or label_vocab is None or char_vocab is None:
            self.token_vocab = Vocab(tokens, min_freq=min_freq, reserved_tokens=reserved_tokens)
            self.label_vocab = Vocab(labels, min_freq=min_freq, reserved_tokens=reserved_tokens)
            self.char_vocab = Vocab(chars, min_freq=min_freq, reserved_tokens=reserved_tokens)
        else:
            self.token_vocab = token_vocab
            self.label_vocab = label_vocab
            self.char_vocab = char_vocab

        # sequences 是还未转化的
        self.sequences, self.valid_len, self.labels = prepareSeq(tokens, labels,
                                                                 self.label_vocab,
                                                                 max_len)
        self.char_len = char_len

    def __getitem__(self, idx):
        seq = self.sequences[idx]  # (T,)
        x = getTokenTensor(seq, self.token_vocab)  # (T,)
        x_char = getCharTensor(seq, self.char_vocab, self.char_len)  # (T, char_len)
        x_len = torch.tensor(self.valid_len[idx])  # (1,)
        y = torch.LongTensor(self.labels[idx])  # (T,)
        return x, x_len, x_char, y

    def __len__(self):
        return len(self.sequences)
