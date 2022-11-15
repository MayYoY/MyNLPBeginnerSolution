import torch
import torchtext
import pandas as pd
from torchtext.legacy import data


def prepareCSV(read_path, target_path):
    lines = open(read_path, encoding="utf-8", mode="r").readlines()
    lines = [line.rstrip('\n') for line in lines]
    df = pd.DataFrame()
    df["Sequence"] = lines
    df.to_csv(target_path, index=False, encoding='utf_8_sig')


def getDataset(train_path, batch_size, max_len=30, eval_path=None):
    def tokenizer(text):
        return list(text)
    text_field = data.Field(sequential=True, init_token="<bos>", eos_token="<eos>",
                            batch_first=True, tokenize=tokenizer, fix_length=max_len)
    train_set = data.TabularDataset(path=train_path, fields=[("Sequence", text_field)],
                                    format="csv", skip_header=True)
    text_field.build_vocab(train_set)
    train_iter = data.BucketIterator(train_set, batch_size, train=True, shuffle=True)
    if eval_path is None:
        return train_iter, text_field.vocab
    else:
        eval_set = data.TabularDataset(path=eval_path, fields=[("Sequence", text_field)],
                                       format="csv", skip_header=True)
        eval_iter = data.BucketIterator(eval_set, batch_size, train=False, shuffle=False)
        return train_iter, text_field.vocab, eval_iter
