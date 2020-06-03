from torch import nn
import torch
import torch.nn.functional as F


class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, num_classes=6, nc=300):
        super(DPCNN, self).__init__()

        self.emb_sz = nc
        self.embeddings = None

        # self.config = config
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1, self.channel_size, (3, nc), stride=1)
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.act_fun = nn.ReLU()
        self.linear_out = nn.Linear(self.channel_size, num_classes)

    def init_embedding(self, vectors, n_tokens, device):
        self.embeddings = nn.Embedding(n_tokens, self.emb_sz).to(device)
        self.embeddings.weight.data.copy_(vectors.to(device))

    def embed(self, data):
        embedded = self.embeddings(data)
        return embedded

    def forward(self, x):
        batch = x.shape[0]

        x = x.unsqueeze(1)

        # Region embedding
        x = self.conv_region_embedding(x)        # [batch_size, channel_size, length, 1]

        x = self.padding_conv(x)                      # pad
        x = self.act_fun(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = self.act_fun(x)
        x = self.conv3(x)

        while x.size()[-2] >= 2:
            x = self._block(x)

        x = x.view(batch, self.channel_size)
        x = self.linear_out(x)

        if x.size(1) == 1:
            x = x.squeeze(1)
        return x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = self.pooling(x)

        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x
