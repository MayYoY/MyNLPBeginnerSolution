import torch


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


def confusedMat(pred, y, res):
    pred = pred.argmax(axis=1)
    for i, pr in enumerate(pred):
        res[y[i], pred[i]] += 1
    return res
