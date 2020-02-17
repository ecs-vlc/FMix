from torch import nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, op_size=6, nl=2, bidirectional=True, emb_sz=300, n_hiddenUnits=100):
        super(LSTM, self).__init__()
        self.n_hidden = n_hiddenUnits
        self.emb_sz = emb_sz
        self.embeddings = None

        #  self.embeddings.weight.requires_grad = False
        self.rnn = nn.LSTM(emb_sz, n_hiddenUnits, num_layers=2, bidirectional=True, dropout=0.2, batch_first=True)
        self.lArr = []
        if bidirectional:
            n_hiddenUnits = 2 * n_hiddenUnits
        self.bn1 = nn.BatchNorm1d(num_features=n_hiddenUnits)
        for i in range(nl):
            if i == 0:
                self.lArr.append(nn.Linear(n_hiddenUnits * 3, n_hiddenUnits))
            else:
                self.lArr.append(nn.Linear(n_hiddenUnits, n_hiddenUnits))
        self.lArr = nn.ModuleList(self.lArr)
        self.l1 = nn.Linear(n_hiddenUnits, op_size)

    def init_embedding(self, vectors, n_tokens, device):
        self.embeddings = nn.Embedding(n_tokens, self.emb_sz).to(device)
        self.embeddings.weight.data.copy_(vectors.to(device))

    def embed(self, data):
        torch.cuda.empty_cache()
        bs = data.shape[0]
        self.h = self.init_hidden(bs)
        embedded = self.embeddings(data)
        return embedded

    def forward(self, embedded):
        bs = embedded.shape[0]
        embedded = nn.Dropout()(embedded)
        #         embedded = pack_padded_sequence(embedded, torch.as_tensor(lengths))
        rnn_out, self.h = self.rnn(embedded, (self.h, self.h))
        #         rnn_out, lengths = pad_packed_sequence(rnn_out,padding_value=1)
        avg_pool = F.adaptive_avg_pool1d(rnn_out.permute(0, 2, 1), 1).view(bs, -1)
        max_pool = F.adaptive_max_pool1d(rnn_out.permute(0, 2, 1), 1).view(bs, -1)
        ipForLinearLayer = torch.cat([avg_pool, max_pool, rnn_out[:, -1]], dim=1)
        for linearlayer in self.lArr:
            outp = linearlayer(ipForLinearLayer)
            ipForLinearLayer = self.bn1(F.relu(outp))
            ipForLinearLayer = nn.Dropout(p=0.6)(ipForLinearLayer)
        outp = self.l1(ipForLinearLayer)
        del embedded;
        del rnn_out;
        del self.h;
        torch.cuda.empty_cache()
        return outp

    def init_hidden(self, batch_size):
        return torch.zeros((4, batch_size, self.n_hidden), device="cuda")
