import numpy as np
import pandas as pd
import collections
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def count_corpus(tokens):
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


# Defined in file: ./chapter_recurrent-neural-networks/text-preprocessing.md
class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = reserved_tokens  # ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}
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
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs


def bag_word(corpus):
    """
    词袋模型
    :param corpus:
    :return:
    """
    # re.sub() 将符合pattern的替换为目标字符
    # 去除特殊标点并转为小写
    corpus = [re.sub('[^A-Za-z]', ' ', line).strip().lower() for line in corpus]
    texts = [sentence.split() for sentence in corpus]
    tokens = [token for sentence in texts for token in sentence]
    return texts, tokens


"""def ngrams(corpus, n=3, min_freq=0):
    
    tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    tokens = [token for sentence in tokens for token in sentence]
    counter = collections.Counter(tokens)
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx_to_token = []
    token_to_idx = {}
    for token, freq in counter:
        if freq < min_freq:
            break
        idx_to_token.append(token)
        token_to_idx[token] = len(idx_to_token) - 1
    return idx_to_token, token_to_idx"""


def read_data(method="BoW"):
    assert method in ["BoW", "N-gram"]
    data = pd.read_csv('train.tsv', sep='\t')  # 带标签, 方便评估
    corpus = data['Phrase'].tolist()  # (156060, 15240)
    y = data['Sentiment'].to_numpy()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    return X, y


class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TextDataset:
    def __init__(self, corpus, y):
        self.texts, tokens = bag_word(corpus)
        self.y = y
        self.vocab = Vocab(tokens)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        text = self.texts[idx]
        x = np.zeros((1, len(self.vocab.token_to_idx)))
        for token in text:
            x[0, self.vocab.token_to_idx[token]] += 1
        return x, self.y[idx]


class BatchSampler:
    def __init__(self, dataset=None, shuffle=False, batch_size=1, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        self.num_data = len(dataset)
        if self.drop_last or (self.num_data % batch_size == 0):
            self.num_samples = self.num_data // batch_size  # 批数
        else:
            self.num_samples = self.num_data // batch_size + 1
        indices = np.arange(self.num_data)
        if self.shuffle:
            np.random.shuffle(indices)
        if self.drop_last:
            indices = indices[:self.num_samples * batch_size]  # 去尾
        self.indices = indices

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        batch_indices = []
        for i in range(self.num_samples):
            if (i + 1) * self.batch_size <= self.num_data:
                for idx in range(i * self.batch_size, (i + 1) * self.batch_size):
                    batch_indices.append(self.indices[idx])
                yield batch_indices
                batch_indices = []
            else:
                for idx in range(i * self.batch_size, self.num_data):
                    batch_indices.append(self.indices[idx])
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices


# 根据sampler生成的索引，从dataset中取数据，并组合成一个batch
class DataLoader:
    def __init__(self, dataset, sampler=BatchSampler, shuffle=False, batch_size=1, drop_last=False):
        self.dataset = dataset
        self.sampler = sampler(dataset, shuffle, batch_size, drop_last)

    def __len__(self):
        return len(self.sampler)

    def __call__(self):
        self.__iter__()

    def __iter__(self):
        for sample_indices in self.sampler:
            data_list = []
            label_list = []
            for indice in sample_indices:
                data, label = self.dataset[indice]
                # print(data.toarray().shape)
                # data_list.append(data.toarray())
                data_list.append(data)  # 自行实现的
                label_list.append(label)
            yield np.vstack(data_list), np.hstack(label_list)
