import os

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchtext.data import Iterator, batch, pool
from tqdm import tqdm


class ToxicDataset(Dataset):
    def __init__(self, dataframe, bert_model='bert-base-cased'):
        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.pad_idx = self.tokenizer.pad_token_id

        self.X = []
        self.Y = []
        for i, (row) in tqdm(dataframe.iterrows()):
            x, y = self.row_to_tensor(self.tokenizer, row)
            self.X.append(x)
            self.Y.append(y)

    @staticmethod
    def row_to_tensor(tokenizer, row):
        tokens = tokenizer.encode(row["comment_text"], add_special_tokens=True, max_length=128)

        x = torch.LongTensor(tokens)
        y = torch.FloatTensor(row[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]])
        return x, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


class NoBatchBucketIterator(Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def create_batches(self):
        if self.sort:
            self.batches = batch(self.data(), self.batch_size,
                                 self.batch_size_fn)
        else:
            self.batches = pool(self.data(), self.batch_size,
                                self.sort_key, self.batch_size_fn,
                                random_shuffler=self.random_shuffler,
                                shuffle=self.shuffle,
                                sort_within_batch=self.sort_within_batch)

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)

                x, y = list(zip(*minibatch))
                x = pad_sequence(x, batch_first=True, padding_value=0)
                y = torch.stack(y)
                yield x, y
            if not self.repeat:
                return


def toxic_bert(args):
    trainset_path = os.path.join(args.dataset_path, 'train.csv')
    test_csv_path = os.path.join(args.dataset_path, 'test.csv')
    test_labels_path = os.path.join(args.dataset_path, 'test_labels.csv')
    testset_path = os.path.join(args.dataset_path, 'test_set.csv')
    if not os.path.isfile(testset_path):
        a = pd.read_csv(test_csv_path)
        b = pd.read_csv(test_labels_path)
        b = b.dropna(axis=1)
        merged = a.merge(b, on='id')
        merged = merged[merged['toxic'] >= 0]
        merged.to_csv(testset_path, index=False)

    train_df = pd.read_csv(trainset_path)
    test_df = pd.read_csv(testset_path)

    train_dataset = ToxicDataset(train_df)
    test_dataset = ToxicDataset(test_df)

    train_loader = NoBatchBucketIterator(dataset=train_dataset, batch_size=args.batch_size,
                                                 sort_key=lambda x: x[0].size(0),
                                                 device=torch.device(args.device), sort_within_batch=True)
    test_loader = NoBatchBucketIterator(dataset=test_dataset, batch_size=args.batch_size,
                                                sort_key=lambda x: x[0].size(0),
                                                device=torch.device(args.device), sort_within_batch=True)

    return train_loader, None, test_loader
