import numpy as np


class Model:
    def __init__(self, num_feature, num_class, lr):
        self.w = np.random.randn(num_feature, num_class)  # F x C
        self.b = np.zeros(num_class)
        self.lr = lr
        self.delta_w = 0
        self.delta_b = 0
        self.train = True

    def forward(self, x, y=None):
        pred = x.dot(self.w) + self.b  # B x C
        if self.train:
            assert y is not None, "Labels are needed when training!"  # B
            real = np.zeros((len(y), pred.shape[1]))  # B x C
            real[range(len(y)), y] = 1
            self.delta_b = real - softmax(pred)  # B x C    y - y_hat
            self.delta_w = x.T.dot(self.delta_b)  # F x C   x^T(y - y_hat)
            self.delta_b = self.delta_b.mean(axis=0)  # C
            self.delta_w = self.delta_w / len(y)  # 求平均
        return pred

    def step(self):
        self.w += self.lr * self.delta_w  # w = w + lr * delta
        self.b += self.lr * self.delta_b
        self.delta_w = 0  # zero_grad
        self.delta_b = 0


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((-1, 1))


def loss_fun(pred, y, mean=True):
    pred = softmax(pred)
    loss = -1 * np.log(pred[range(len(y)), y] + 1e-8).sum()
    if mean:
        return loss / len(y)
    else:
        return loss


def correctCount(pred, y):
    pred = pred.argmax(axis=1)
    cnt = np.sum((pred == y).astype(np.float32))
    return cnt


def accuracy(pred, y):
    pred = pred.argmax(axis=1)
    acc = np.sum((pred == y).astype(np.float32)) / pred.shape[0]
    return acc


def evaluate(model, x, y):
    """
    :param model: 模型
    :param x:
    :param y:
    :return:
    """
    model.train = False
    pred = model.forward(x)
    acc = accuracy(pred, y)
    loss = loss_fun(pred, y)
    print(f'accuracy: {acc:.2f}\taverage loss: {loss:.2f}')
