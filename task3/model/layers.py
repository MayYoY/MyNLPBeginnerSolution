import torch.nn as nn
import torch.nn.functional as F
from .utils import sort_by_seq_lens, masked_softmax, weighted_sum


class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.
    对同一序列内的元素作相同的dropout
    """

    def forward(self, sequences_batch):
        """
        sequences_batch: B x T x embed_size
        """
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = F.dropout(ones, self.p, self.training, inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    def __init__(self, rnn_type, input_size, hidden_size,
                 num_layers=1, bias=True, dropout=0.0, bidirectional=False):
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn = rnn_type(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        # B x T x dim
        sorted_batch, sorted_lengths, _, restoration_idx = sort_by_seq_lens(sequences_batch,
                                                                            sequences_lengths)
        # 打包padded的序列, 方便RNN处理
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths.cpu(),
                                                         batch_first=True)

        outputs, _ = self.rnn(packed_batch, None)

        # 复原序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class SoftmaxAttention(nn.Module):
    """
    交互注意力输出
    Attention layer taking premises and hypotheses encoded by an RNN as input
    and computing the soft attention between their elements.
    """

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        """
        Args:
            premise_batch: B x T x dim
            premise_mask:
            hypothesis_batch: B x T x dim
            hypothesis_mask:
        """
        # e_{ij} = a_i^T \dot b_j
        # 简化为矩阵相乘, 计算注意力权重
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1).contiguous())
        # 带掩码的 softmax
        # B x T(L_a) x T(L_b)
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        # B x T(L_b) x T(L_a)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2).contiguous(),
                                       premise_mask)

        # hypothesis, premise相互之间的注意力输出
        # a_i = \sum^{L_b} \alpha_j b_j
        # b_j = \sum^{L_a} \alpha_i a_i
        # B x T(L_a) x 2H(hidden_size)
        attended_premises = weighted_sum(hypothesis_batch,  # B x T_b x d
                                         prem_hyp_attn,  # B x T(L_a) x T(L_b)
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses
