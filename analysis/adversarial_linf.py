# !pip install foolbox
import torch
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np
import json

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn

from sklearn.model_selection import ParameterGrid

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Imagenet Training')
parser.add_argument('--arr', default=0, type=int, help='point in job array')
args = parser.parse_args()

param_grid = ParameterGrid({
    'mode': ('baseline', 'cutmix', 'mixup', 'fmix'),
    'repeat': list(range(5))
})

params = param_grid[args.arr]
mode = params['mode']
repeat = params['repeat']

test_transform = transforms.Compose([
    transforms.ToTensor()  # convert to tensor
])

# load data
testset = CIFAR10(".", train=False, download=True, transform=test_transform)
testloader = DataLoader(testset, batch_size=750, shuffle=False, num_workers=5)

attacks = [
    fa.FGSM(),
    fa.LinfPGD(),
    fa.LinfBasicIterativeAttack(),
    fa.LinfAdditiveUniformNoiseAttack(),
    fa.LinfDeepFoolAttack(),
]

epsilons = [
    0.0,
    0.0005,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.1,
    0.3,
    0.5,
    1.0,
]


def normalize_with(mean, std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    return lambda x: (x - mean.to(x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)) / std.to(x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

# (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)


class Normalized(nn.Module):
    def __init__(self, model, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
        super().__init__()
        self.model = model
        self.normalize = normalize_with(mean=mean, std=std)

    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)


results = dict()

# for mode in ('baseline', 'mixup', 'fmix', 'cutmix', 'fmixplusmixup'):
print(mode)
results[mode] = dict()

model = Normalized(torch.hub.load('ecs-vlc/FMix:master', f'preact_resnet18_cifar10_{mode}', pretrained=True, repeat=repeat))
model.eval()
fmodel = PyTorchModel(model, bounds=(0, 1))

attack_success = np.zeros((len(attacks), len(epsilons), len(testset)), dtype=np.bool)
for i, attack in enumerate(attacks):
    # print(attack)
    idx = 0
    for images, labels in tqdm(testloader):
        # print('.', end='')
        images = images.to(fmodel.device)
        labels = labels.to(fmodel.device)

        _, _, success = attack(fmodel, images, labels, epsilons=epsilons)
        success_ = success.cpu().numpy()
        attack_success[i][:, idx:idx + len(labels)] = success_
        idx = idx + len(labels)
    # print("")

import pickle
with open(f'adversarial_linf_{mode}_{repeat}.p', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(attack_success, f)
# for i, attack in enumerate(attacks):
#     results[mode][str(attack)] = (1.0 - attack_success[i].mean(axis=-1)).tolist()
#
# robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
# results[mode]['robust_accuracy'] = robust_accuracy.tolist()
#
# with open('adv-results-cifar-linf.json', 'w') as fp:
#     json.dump(results, fp)
