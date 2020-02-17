import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchbearer import state_key
from torchbearer import callbacks

T = state_key('t')
T_SHUFFLED = state_key('t_shuffled')
MI = state_key('mi')


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DoNothing(nn.Module):
    def forward(self, x):
        return x


def resample(x):
    return F.fold(F.unfold(x, kernel_size=2, stride=2), (int(x.size(2) / 2), int(x.size(3) / 2)), 1)


class Estimator(nn.Module):
    def __init__(self, conv, in_size, pool_input=False, halves=0):
        super().__init__()
        self.pool = DoNothing()
        self.halves = halves

        if conv:
            in_size = in_size + 3 * 4 ** halves
            if pool_input:
                self.pool = nn.AdaptiveAvgPool2d((8, 8))

            self.est = nn.Sequential(
                nn.Conv2d(in_size, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                Flatten(),
                nn.Linear(2 * 2 * 256, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            in_size = in_size + 32 * 32 * 3
            self.est = nn.Sequential(
                nn.Linear(in_size, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

    def forward(self, x, f):
        if self.halves < 5:
            for i in range(self.halves):
                x = resample(x)
        else:
            x = x.view(x.size(0), -1)
            f = f.view(f.size(0), -1)
        if f.dim() == 2:
            x = x.view(x.size(0), -1)
        x = torch.cat((x, f), dim=1)
        x = self.pool(x)

        return self.est(x)


cfgs = {
    'A': {
        'f1': lambda: Estimator(True, 64, halves=1), 'f2': lambda: Estimator(True, 128, halves=2), 'f3': lambda: Estimator(True, 256, halves=2),
        'f4': lambda: Estimator(True, 256, halves=3), 'f5': lambda: Estimator(True, 512, halves=3), 'f6': lambda: Estimator(True, 512, halves=4),
        'f7': lambda: Estimator(True, 512, halves=4),
        'f8': lambda: Estimator(False, 512, False, halves=5), 'c1': lambda: Estimator(False, 2048), 'c2': lambda: Estimator(False, 2048)},
    'B': {},
    'D': {},
    'E': {},
}


def mi(tanh):
    def mi_loss(state):
        m_t, m_t_shuffled = state[torchbearer.Y_PRED]
        mi = {}
        sum = 0.0
        for layer in m_t.keys():
            t = m_t[layer]
            t_shuffled = m_t_shuffled[layer]
            if tanh:
                t = t.tanh()
                t_shuffled = t_shuffled.tanh()
            tmp = t.mean() - (torch.logsumexp(t_shuffled, 0) - math.log(t_shuffled.size(0)))
            mi[layer] = tmp.item()
            sum += tmp
        if len(mi.keys()) == 1:
            state[MI] = mi[next(iter(mi.keys()))]
        else:
            state[MI] = mi
        return -sum
    return mi_loss


def process(x, cache, cfg):
    t = {}
    t_shuffled = {}

    for layer in cfg.keys():
        out = cache[layer].detach()
        t[layer] = cfg[layer](x, out)
        t_shuffled[layer] = cfg[layer](x, out[torch.randperm(out.size(0))])
    return t, t_shuffled


class MimeVGG(nn.Module):
    def __init__(self, vgg, cfg):
        super().__init__()

        self.vgg = vgg
        self.cfg = nn.ModuleDict(cfg)

    def forward(self, x):
        pred, cache = self.vgg(x)

        t, t_shuffled = process(x, cache, self.cfg)

        return t, t_shuffled


if __name__ == '__main__':
    from torch import optim
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    import torchbearer
    from torchbearer import Trial

    OTHER_MI = state_key('other_mi')

    cfg = cfgs['A']

    import argparse

    parser = argparse.ArgumentParser(description='VGG MI')
    parser.add_argument('--model', default='mix_3', type=str, help='model')
    args = parser.parse_args()

    from .vgg import vgg11_bn

    vgg = vgg11_bn(return_cache=True)
    vgg.load_state_dict(torch.load(args.model + '.pt')[torchbearer.MODEL])
    for param in vgg.parameters():
        param.requires_grad = False

    for layer in cfg:
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_base = [transforms.ToTensor(), normalize]

        transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + transform_base

        transform_train = transforms.Compose(transform)
        transform_test = transforms.Compose(transform_base)

        trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        valset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True, num_workers=8)
        valloader = torch.utils.data.DataLoader(valset, batch_size=5000, shuffle=True, num_workers=8)

        model = MimeVGG(vgg, {k: cfgs['A'][k]() for k in [layer]})

        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=5e-4)

        mi_false = mi(False)

        @callbacks.add_to_loss
        def mi_no_tanh(state):
            state[OTHER_MI] = mi_false(state)
            return 0

        trial = Trial(model, optimizer, mi(True), metrics=['loss', torchbearer.metrics.mean(OTHER_MI)], callbacks=[mi_no_tanh, callbacks.TensorBoard(write_graph=False, comment='mi_' + args.model, log_dir='mi_data')])
        trial.with_generators(train_generator=trainloader, val_generator=valloader).to('cuda')
        trial.run(20, verbose=1)
