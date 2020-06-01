from torch import nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, num_classes=6, nl=2, bidirectional=True, nc=300, hidden_sz=128):
        super(LSTM, self).__init__()
        self.hidden_sz = hidden_sz
        self.emb_sz = nc
        self.embeddings = None

        self.rnn = nn.LSTM(nc, hidden_sz, num_layers=2, bidirectional=bidirectional, dropout=0, batch_first=True)
        if bidirectional:
            hidden_sz = 2 * hidden_sz

        layers = []
        for i in range(nl):
            if i == 0:
                layers.append(nn.Linear(hidden_sz * 3, hidden_sz))
            else:
                layers.append(nn.Linear(hidden_sz, hidden_sz))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_sz, num_classes)

    def init_embedding(self, vectors, n_tokens, device):
        self.embeddings = nn.Embedding(n_tokens, self.emb_sz).to(device)
        self.embeddings.weight.data.copy_(vectors.to(device))

    def embed(self, data):
        self.h = self.init_hidden(data.size(0))
        embedded = self.embeddings(data)
        return embedded

    def forward(self, embedded):
        rnn_out, self.h = self.rnn(embedded, (self.h, self.h))

        avg_pool = F.adaptive_avg_pool1d(rnn_out.permute(0, 2, 1), 1).view(embedded.size(0), -1)
        max_pool = F.adaptive_max_pool1d(rnn_out.permute(0, 2, 1), 1).view(embedded.size(0), -1)
        x = torch.cat([avg_pool, max_pool, rnn_out[:, -1]], dim=1)
        x = self.layers(x)
        res = self.output(x)
        if res.size(1) == 1:
            res = res.squeeze(1)
        return res

    def init_hidden(self, batch_size):
        return torch.zeros((4, batch_size, self.hidden_sz), device="cuda")
