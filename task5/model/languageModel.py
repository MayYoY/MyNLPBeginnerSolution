import torch
from torch import nn
import torch.nn.functional as F
from . import metric
from dataset import loader
import d2l.torch as d2l
from tqdm.auto import tqdm
import math
import numpy as np


class MyLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, rnn_type="LSTM"):
        super(MyLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(hidden_size=hidden_size, input_size=embed_size, dropout=0.5)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(hidden_size=hidden_size, input_size=embed_size, dropout=0.5)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(hidden_size=hidden_size, input_size=embed_size, dropout=0.5)
        else:
            print("Unsupported type of rnn! Building LSTM instead.")
            self.rnn = nn.LSTM(hidden_size=hidden_size, input_size=embed_size, dropout=0.5)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, state=None):
        b, t = x.shape  # B x T
        x = self.embed(x).permute(1, 0, 2)
        x, out_state = self.rnn(x, state)  # output, state
        x = x.permute(1, 0, 2).reshape(-1, x.shape[-1])
        y = self.fc(x).reshape(b, t, -1)  # B x T x vocab_size
        return y, out_state

    def begin_state(self, batch_size, device):
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((1, batch_size, self.hidden_size), device=device)
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((1, batch_size, self.hidden_size), device=device),
                    torch.zeros((1, batch_size, self.hidden_size), device=device))

    def init_weight(self, m):
        if type(m) in [nn.Linear, nn.Embedding]:
            nn.init.xavier_uniform_(m.weight)
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])


def decode_pred(vocab, pred):
    """
    解码预测, 用于计算 BLEU
    :param vocab:
    :param pred:
    :return:
    """
    pass


def train(net, loss_fun, optimizer, num_epochs, train_iter, eval_iter, device):
    net.to(device)
    print("Training...")
    progress_bar = tqdm(range(num_epochs * len(train_iter)))
    metric_helper = metric.Accumulator()
    for i in range(num_epochs):
        print(f"Epoch{i + 1}:")
        net.train()
        metric_helper.reset()
        for seq, valid_len in train_iter:
            optimizer.zero_grad()  # 清零
            x, y = seq[:, :-1].to(device), seq[:, 1:].to(device)  # B x T'
            valid_len = valid_len.to(device)
            pred, _ = net(x)  # B x T' x vocab_size
            loss = loss_fun(pred, y, valid_len)
            loss.sum().backward()  # 反馈
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.)  # 截断
            optimizer.step()  # 更新
            progress_bar.update(1)
            metric_helper.update(seq.shape[0], loss.sum())
        print(f'train stage: average loss {metric_helper.getAvg():.4f}, '
              f'perplexity {metric_helper.getPPL():.4f}')
        eval_loss, eval_ppl = evaluate(net, loss_fun, eval_iter, device)
        print(f'evaluation stage: average loss {eval_loss:.4f}, perplexity {eval_ppl:.4f}')


def trainForText(net, loss_fun, optimizer, num_epochs, train_iter, eval_iter, device):
    net.to(device)
    print("Training...")
    progress_bar = tqdm(range(num_epochs * len(train_iter)))
    metric_helper = metric.Accumulator()
    for i in range(num_epochs):
        print(f"Epoch{i + 1}:")
        net.train()
        metric_helper.reset()
        for batch in train_iter:
            optimizer.zero_grad()  # 清零
            seq = batch.Sequence
            x, y = seq[:, :-1].to(device), seq[:, 1:].to(device)  # B x T'
            begin_state = net.begin_state(seq.shape[0], device)
            pred, _ = net(x, begin_state)  # B x T' x vocab_size
            pred = pred.view(-1, pred.shape[-1])
            y = y.flatten()
            # loss = loss_fun(pred, y)  # reduction = "none"
            # loss = torch.masked_select(loss, y != 1).sum()  # vocab["<pad>"] = 1
            loss = loss_fun(pred, y)  # reduction = "mean"
            loss.backward()  # 反馈
            # nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.)  # 截断
            optimizer.step()  # 更新
            progress_bar.update(1)
            metric_helper.update(seq.shape[0], loss)
        print(f'train stage: average loss {metric_helper.total_loss:.4f}, '
              f'perplexity {math.exp(metric_helper.total_loss):.4f}')
        # eval_loss, eval_ppl = evaluate(net, loss_fun, eval_iter, device)
        # print(f'evaluation stage: average loss {eval_loss:.4f}, perplexity {eval_ppl:.4f}')


def evaluate(net, loss_fun, eval_iter, device):
    net.eval()
    metric_helper = metric.Accumulator()
    for seq, valid_len in eval_iter:
        x, y = seq[:, :-1].to(device), seq[:, 1:].to(device)
        valid_len = valid_len.to(device)
        pred, _ = net(x)  # B x T' x vocab_size
        loss = loss_fun(pred, y, valid_len)
        metric_helper.update(seq.shape[0], loss.sum())
    return metric_helper.getAvg(), metric_helper.getPPL()


def generate(net, input_seq, vocab, device, num_step=-1):
    """
    根据输入序列生成预测诗句
    :param net:
    :param input_seq: str, 输入序列
    :param vocab: 词典
    :param device
    :param num_step: int, 若为 -1, 则遇到 <eos> 才停止
    :return:
    """
    net.eval()
    net.to(device)
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[input_seq[0]]]
    get_input = lambda: torch.tensor(outputs[-1], dtype=torch.int64,
                                     device=device).reshape(1, 1)  # B x T
    # Warm-up period
    for y in input_seq[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # predict
    if num_step == -1:
        while True:
            y, state = net(get_input(), state)  # y: B x T x vocab_size
            pred_idx = int(y.argmax(dim=2).reshape(1))
            if pred_idx == vocab["<eos>"] or len(outputs) > 175:
                break
            outputs.append(pred_idx)
    else:
        for _ in range(num_step):
            y, state = net(get_input(), state)  # y: B x T x vocab_size
            pred_idx = int(y.argmax(dim=2).reshape(1))
            if pred_idx == vocab["<eos>"]:
                break
            outputs.append(pred_idx)
    if isinstance(vocab, loader.Vocab):
        output_seq = ''.join([vocab.idx_to_token[i] for i in outputs])
    else:
        output_seq = ''.join([vocab.itos[i] for i in outputs])
    return output_seq
