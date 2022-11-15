import torch
from torch.utils.data import Dataset
import re
import os
from . import preprocess


def read_snli(path):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    with open(path, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels


class SNLIDataSet(Dataset):
    def __init__(self, text_data, max_len=None):
        # 样本数
        self.num_sequence = len(text_data[0])
        self.premises_len = [len(seq) for seq in text_data[0]]  # valid_lens
        self.hypothesis_len = [len(seq) for seq in text_data[1]]

        premises = preprocess.tokenize(text_data[0])
        hypotheses = preprocess.tokenize(text_data[1])
        tokens = premises + hypotheses
        # 预留pad，去除长尾数据
        self.vocab = preprocess.Vocab(tokens, min_freq=5, reserved_tokens=["<pad>"])

        # max_len for padding
        if max_len is None:
            self.max_premises_len = max(self.premises_len)
            self.max_hypothesis_len = max(self.hypothesis_len)
        else:
            self.max_premises_len = max_len
            self.max_hypothesis_len = max_len

        premises = self._pad(premises, self.max_premises_len)
        hypotheses = self._pad(hypotheses, self.max_hypothesis_len)

        # padding
        self.data = {"premises": premises, "hypothesis": hypotheses,
                     "labels": torch.tensor(text_data[2])}

    def __len__(self):
        return self.num_sequence

    def _pad(self, lines, max_len):
        return torch.tensor([preprocess.truncate_pad(
            self.vocab[line], max_len, self.vocab['<pad>'])
            for line in lines])

    def __getitem__(self, index):
        return {"premises": self.data["premises"][index],
                "premises_len": min(self.premises_len[index], self.max_premises_len),  # 可能被截断
                "hypothesis": self.data["hypothesis"][index],
                "hypothesis_len": min(self.hypothesis_len[index], self.max_hypothesis_len),
                "labels": self.data["labels"][index]}
