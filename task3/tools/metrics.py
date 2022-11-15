import torch
import re


def tokenizer(corpus):
    corpus = [re.sub('[^A-Za-z]', ' ', line).strip().lower() for line in corpus]
    texts = [sentence.split() for sentence in corpus]
    tokens = [token for sentence in texts for token in sentence]
    return texts, tokens


class Accumulate:
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.val = 0
        self.cnt = 0

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.val = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.cnt += n
        self.val = val
        self.sum += val
        self.avg = self.sum / self.cnt


def correctCount(pred, y, acc=False):
    pred = pred.argmax(axis=1)
    cnt = torch.sum((pred == y), dtype=torch.float)
    if acc:
        cnt /= pred.shape[0]
    return cnt
