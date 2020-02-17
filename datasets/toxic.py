import spacy
import re
from torchtext import data
import torchtext
import pandas as pd
import os
import torch


def toxic_ds(args):
    tok = spacy.load('en')
    stopwords = spacy.lang.en.stop_words.STOP_WORDS
    stopwords.update(['wikipedia', 'article', 'articles', 'im', 'page'])

    def spacy_tok(x):
        x = re.sub(r'[^a-zA-Z\s]', '', x)
        x = re.sub(r'[\n]', ' ', x)
        return [t.text for t in tok.tokenizer(x)]

    TEXT = data.Field(lower=True, tokenize=spacy_tok, eos_token='EOS', stop_words=stopwords, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
    dataFields = [("id", None), ("comment_text", TEXT), ("toxic", LABEL), ("severe_toxic", LABEL), ("threat", LABEL),
                  ("obscene", LABEL), ("insult", LABEL), ("identity_hate", LABEL)]
    trainset_path = os.path.join(args.dataset_path, 'train.csv')
    dataset = data.TabularDataset(path=trainset_path, format='csv', fields=dataFields, skip_header=True)
    train, val = dataset.split()

    TEXT.build_vocab(train, vectors='fasttext.simple.300d')
    traindl, valdl = torchtext.data.BucketIterator.splits(datasets=(train, val),
                                                          batch_sizes=(args.batch_size, args.batch_size),
                                                          sort_key=lambda x: len(x.comment_text),
                                                          device=torch.device(args.device),
                                                          sort_within_batch=True
                                                          )

    test_csv_path = os.path.join(args.dataset_path, 'test.csv')
    test_labels_path = os.path.join(args.dataset_path, 'test_labels.csv')
    test_set_path = os.path.join(args.dataset_path, 'test_set.csv')
    if not os.path.isfile(test_set_path):
        a = pd.read_csv(test_csv_path)
        b = pd.read_csv(test_labels_path)
        b = b.dropna(axis=1)
        merged = a.merge(b, on='id')
        merged = merged[merged['toxic'] >= 0]
        merged.to_csv(test_set_path, index=False)

    testset = data.TabularDataset(path=test_set_path, format='csv', fields=dataFields, skip_header=True)
    testdl = torchtext.data.BucketIterator(dataset=testset, batch_size=64, sort_key=lambda x: len(x.comment_text),
                                           device=torch.device(args.device), sort_within_batch=True)
    vectors = train.fields['comment_text'].vocab.vectors.to(args.device)
    traindl, valdl, testdl = BatchGenerator(traindl), BatchGenerator(valdl), BatchGenerator(testdl)
    traindl.vectors = vectors
    traindl.ntokens = len(TEXT.vocab)

    return traindl, valdl, testdl


class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.yFields = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        self.x = 'comment_text'

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            X = list(getattr(batch, self.x))
            X[0] = X[0].permute(1, 0)
            y = torch.transpose(torch.stack([getattr(batch, y) for y in self.yFields]), 0, 1)
            yield (X, y)


from torchbearer import Callback
import torchbearer


class ToxicHelper(Callback):
    def on_start(self, state):
        super().on_start(state)
        vectors = state[torchbearer.TRAIN_GENERATOR].vectors
        ntokens = state[torchbearer.TRAIN_GENERATOR].ntokens
        state[torchbearer.MODEL].init_embedding(vectors, ntokens, state[torchbearer.DEVICE])

    def on_sample(self, state):
        state[torchbearer.X], lengths = state[torchbearer.X]
        state[torchbearer.Y_TRUE] = state[torchbearer.Y_TRUE].float()

        state[torchbearer.X] = state[torchbearer.MODEL].embed(state[torchbearer.X])

    def on_sample_validation(self, state):
        state[torchbearer.X], lengths = state[torchbearer.X]
        state[torchbearer.Y_TRUE] = state[torchbearer.Y_TRUE].float()

        state[torchbearer.X] = state[torchbearer.MODEL].embed(state[torchbearer.X])




