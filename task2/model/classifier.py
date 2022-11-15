import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils import metrics


class MyCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class,
                 method="random", num_map=100, kernel_sizes=None,
                 vectors=None, freeze=True, pad_idx=None):
        super(MyCNN, self).__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5]
        assert len(kernel_sizes) == 3, "Only 3 kernels are supported!"
        assert method in ["glove", "random"], "Unsupported embedding method!"
        if method == "glove":
            assert vectors is not None and pad_idx is not None
            self.embedding = nn.Embedding(vocab_size, embed_size,
                                          pad_idx).from_pretrained(vectors, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        # 多个卷积核
        self.conv1 = nn.Conv2d(1, num_map, (kernel_sizes[0], embed_size))
        self.conv2 = nn.Conv2d(1, num_map, (kernel_sizes[1], embed_size))
        self.conv3 = nn.Conv2d(1, num_map, (kernel_sizes[2], embed_size))

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_map * 3, num_class)

    def forward(self, x):
        x = self.embedding(x)  # B x N -> B x N x embed_size
        x = torch.unsqueeze(x, dim=1)  # B x 1 x N x embed_size

        f1 = F.relu(self.conv1(x))  # B x C x H x 1
        f2 = F.relu(self.conv2(x))  # relu!!!
        f3 = F.relu(self.conv3(x))

        # B x C x 1 x 1 -> B x C
        f1 = F.max_pool1d(f1.squeeze(3), f1.shape[2]).squeeze(2)
        f2 = F.max_pool1d(f2.squeeze(3), f2.shape[2]).squeeze(2)
        f3 = F.max_pool1d(f3.squeeze(3), f3.shape[2]).squeeze(2)

        y = torch.cat([f1, f2, f3], dim=1)  # B x 3C
        y = self.dropout(y)
        y = self.fc(y)

        return y


class MyRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_class, num_hidden,
                 method="random", vectors=None, freeze=True, pad_idx=None):
        super(MyRNN, self).__init__()
        assert method in ["glove", "random"], "Unsupported embedding method!"
        if method == "glove":
            assert vectors is not None and pad_idx is not None
            self.embedding = nn.Embedding(vocab_size, embed_size,
                                          pad_idx).from_pretrained(vectors, freeze=freeze)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)

        # 双向LSTM
        self.rnn = nn.LSTM(input_size=embed_size, hidden_size=num_hidden, bidirectional=True)

        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_hidden * 4, num_class)

    def forward(self, x):
        x = self.embedding(x)  # B x N -> B x N x embed_size
        x = x.permute(1, 0, 2)  # N x B x embed_size

        # self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # N x B x H
        x = torch.cat([x[0], x[-1]], dim=1)

        y = self.fc(self.dropout(x))
        return y


def train(model: nn.Module, train_iter, eval_iter,
          num_epochs, optimizer, loss_fun, device, num_class):
    model = model.to(device)
    progress_bar = tqdm(range(num_epochs * len(train_iter)))
    record_loss = metrics.Accumulate()
    record_acc = metrics.Accumulate()
    for epoch in range(num_epochs):
        model.train()
        record_loss.reset()
        record_acc.reset()
        for batch in train_iter:
            optimizer.zero_grad()

            x = batch.Phrase.to(device)
            y = batch.Sentiment.to(device)
            pred = model(x)

            loss = loss_fun(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # 梯度截断
            optimizer.step()

            progress_bar.update(1)
            record_loss.update(loss)
            record_acc.update(metrics.correctCount(pred, y), len(y))
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1}: average loss: {record_loss.avg:.3f}\taccuracy: {record_acc.avg:.3f}')
        if (epoch + 1) % 10 == 0:
            evaluate(model, eval_iter, loss_fun, device, num_class)


def evaluate(model: nn.Module, eval_iter, loss_fun, device, num_class):
    model.to(device)
    record_loss = metrics.Accumulate()
    record_acc = metrics.Accumulate()
    confused_mat = torch.zeros((num_class, num_class), device=device)
    model.eval()
    for batch in eval_iter:
        x = batch.Phrase.to(device)
        y = batch.Sentiment.to(device)
        pred = model(x)
        loss = loss_fun(pred, y)

        record_loss.update(loss)
        record_acc.update(metrics.correctCount(pred, y), len(y))
        confused_mat += metrics.confusedMat(pred, y, confused_mat)
    print(f'Evaluation: average loss: {record_loss.avg:.3f}\taccuracy: {record_acc.avg:.3f}')
    return confused_mat
