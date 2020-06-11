"""
ImageNet models trained with the original imagenet_hdf5 data set have their outputs in the wrong order. This loads a
model and re-orders the output weights to be consistent with other ImageNet models.
"""
import argparse
import os
from torchvision.models import resnet101
import torch
import torchbearer
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default=None, help='ImageNet path')
parser.add_argument('--model-file')
args = parser.parse_args()

root = os.path.join(args.dataset_path, 'train')

old_classes = list(filter(lambda f: '.hdf5' in f, os.listdir(root)))
new_classes = sorted(old_classes)

model = nn.DataParallel(resnet101(False))
sd = torch.load(args.model_file, map_location='cpu')[torchbearer.MODEL]
model.load_state_dict(sd)
model = model.module

new_weights = torch.zeros_like(model.fc.weight.data)
new_bias = torch.zeros_like(model.fc.bias.data)

for layer in range(1000):
    new_layer = new_classes.index(old_classes[layer])

    new_weights[layer, :] = model.fc.weight[layer, :]
    new_bias[layer] = model.fc.bias[layer]

model.fc.weight.data = new_weights.data
model.fc.bias.data = new_bias.data

torch.save(model.state_dict(), args.model_file + '_converted.pt')
