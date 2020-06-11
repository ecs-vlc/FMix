from torch import nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=6, nl=2, nc=300, hidden_sz=128):
        super(CNN, self).__init__()
        self.hidden_sz = hidden_sz
        self.emb_sz = nc
        self.embeddings = None

        self.conv = nn.Sequential(
            nn.Conv1d(nc, hidden_sz, 3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden_sz, hidden_sz, 3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden_sz, hidden_sz, 3, padding=1),
            nn.ReLU()
        )

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
        embedded = self.embeddings(data)
        return embedded

    def forward(self, embedded):
        x = self.conv(embedded.permute(0, 2, 1))

        avg_pool = F.adaptive_avg_pool1d(x, 1).view(embedded.size(0), -1)
        max_pool = F.adaptive_max_pool1d(x, 1).view(embedded.size(0), -1)
        x = torch.cat([avg_pool, max_pool, x.permute(0, 2, 1)[:, -1]], dim=1)
        x = self.layers(x)
        res = self.output(x)
        if res.size(1) == 1:
            res = res.squeeze(1)
        return res
