import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_mask(X, valid_len):
    """
    在序列中屏蔽不相关的项
    :param X: B x T
    :param valid_len: B x 1
    :return mask: B x T
    """
    max_len = torch.max(valid_len)  # T
    # None, : 和 :, None 是维度扩充, 将 tensor 变为 1 x T 和 B x 1
    mask = torch.arange(max_len, dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # X[~mask] = value
    return mask.float()


def get_mask(sequences_batch, sequences_lengths):
    """
    Get the mask for a batch of padded variable length sequences.

    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).

    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float, device=sequences_batch.device)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    :param tensor: B x * x T
    :param mask: B x T
    """
    tensor_shape = tensor.size()  # B x * x T
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 防止除零
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


def sort_by_seq_lens(batch, valid_lens, descending=True):
    """
    根据句子的有效长度进行排序
    """
    sorted_seq_lens, sorting_index = valid_lens.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)  # 对batch排序
    # [0, 1,..., valid_lens]
    idx_range = valid_lens.new_tensor(torch.arange(0, len(valid_lens)),
                                      device=sorted_batch.device)
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def weighted_sum(tensor, weights, mask):
    """
    加权注意力输出
    :param tensor: B x T_b x 2H or B x T_a x 2H
    :param weights: B x T_a x T_b or B x T_b x T_a
    :param mask: B x T
    """
    weighted_sum = weights.bmm(tensor)  # B x T_a x 2H

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)  # B x T x 1
    mask = mask.expand_as(weighted_sum).contiguous().float()  # .to(weighted_sum.device)

    return weighted_sum * mask


def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.
    :param tensor: B x T x d
    :param mask: B x T
    :param value:
    """
    mask = mask.unsqueeze(1).transpose(2, 1)  # B x T x 1
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add
