import torch
import torch.nn as nn
from tqdm.auto import tqdm
from tools import metrics

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention
from .utils import sequence_mask, replace_masked, get_mask


class ESIM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 vectors=None, pad_idx=0, dropout=0.5, num_classes=3):
        """
        :param vocab_size:
        :param embed_size:
        :param hidden_size:
        :param vectors: 预训练词向量
        :param pad_idx:
        :param dropout:
        :param num_classes:
        """
        super(ESIM, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self._word_embedding = nn.Embedding(vocab_size,
                                            self.embed_size,
                                            padding_idx=pad_idx,
                                            _weight=vectors)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM, self.embed_size,
                                        self.hidden_size, bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(8 * self.hidden_size, self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM, self.hidden_size,
                                           self.hidden_size, bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(8 * self.hidden_size, self.hidden_size),
                                             nn.Tanh(), nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size, num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypothesis,
                hypotheses_lengths):
        """
        :param premises: B x T
        :param premises_lengths: B,
        :param hypothesis: B x T
        :param hypotheses_lengths: B,
        """
        premises_mask = sequence_mask(premises, premises_lengths)  # B x T
        hypotheses_mask = sequence_mask(hypothesis, hypotheses_lengths)
        """premises_mask = get_mask(premises, premises_lengths)
        hypotheses_mask = get_mask(hypothesis, hypotheses_lengths)"""

        embedded_premises = self._word_embedding(premises)  # B x T x embed_size
        embedded_hypotheses = self._word_embedding(hypothesis)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)
        # BLSTM
        encoded_premises = self._encoding(embedded_premises, premises_lengths)  # B x T x 2H
        encoded_hypotheses = self._encoding(embedded_hypotheses, hypotheses_lengths)
        # 注意力操作
        attended_premises, attended_hypotheses = self._attention(encoded_premises,
                                                                 premises_mask,
                                                                 encoded_hypotheses,
                                                                 hypotheses_mask)
        # 拼接, 高级语义和低级语义结合
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)  # B x T x 8H
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)
        # 隐藏层投影
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)
        # 再次输入 BLSTM
        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        # avg && max
        premises_mask = premises_mask.to(v_ai.device)
        hypotheses_mask = hypotheses_mask.to(v_bj.device)
        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1).transpose(2, 1),
                            dim=1) / torch.sum(premises_mask,
                                               dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1).transpose(2, 1),
                            dim=1) / torch.sum(hypotheses_mask,
                                               dim=1, keepdim=True)
        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)
        pred = self._classification(v)
        return pred


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if module.bidirectional:
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0


def train(model: ESIM, train_iter, num_epochs, optimizer, loss_fun, device, eval_iter):
    progress_bar = tqdm(range(num_epochs * len(train_iter)))
    record_loss = metrics.Accumulate()
    model.train()
    record_acc = metrics.Accumulate()
    for epoch in range(num_epochs):
        record_loss.reset()
        record_acc.reset()
        for batch in train_iter:
            optimizer.zero_grad()

            premises = batch["premises"].to(device)
            premises_len = batch["premises_len"].to(device)
            hypothesis = batch["hypothesis"].to(device)
            hypothesis_len = batch["hypothesis_len"].to(device)
            labels = batch["labels"].to(device)

            pred = model(premises, premises_len, hypothesis, hypothesis_len)

            loss = loss_fun(pred, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            progress_bar.update(1)
            record_loss.update(loss)
            record_acc.update(metrics.correctCount(pred, labels), len(premises))
        print(f'Epoch {epoch + 1}: average loss: {record_loss.avg:.3f}\t'
              f'accuracy: {record_acc.avg:.3f}')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"./esim_checkpoint{epoch + 1}")
    evaluate(model, eval_iter, loss_fun, device)


def evaluate(model, eval_iter, loss_fun, device):
    progress_bar = tqdm(range(len(eval_iter)))
    record_loss = metrics.Accumulate()
    record_acc = metrics.Accumulate()
    model.to(device)
    model.eval()
    for batch in eval_iter:
        premises = batch["premises"].to(device)
        premises_len = batch["premises_len"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        hypothesis_len = batch["hypothesis_len"].to(device)
        labels = batch["labels"].to(device)

        pred = model(premises, premises_len, hypothesis, hypothesis_len)
        loss = loss_fun(pred, labels)

        progress_bar.update(1)
        record_loss.update(loss)
        record_acc.update(metrics.correctCount(pred, labels), len(premises))
    print(f'Evaluation: average loss: {record_loss.avg:.3f}\t'
          f'accuracy: {record_acc.avg:.3f}')
