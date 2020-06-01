"""
ImageNet models trained with the imagenet_hdf5 data set have their outputs in the wrong order. This loads a model and
re-orders the output weights to be consistent with other ImageNet models.
"""
import argparse
import os
from torchvision.datasets import ImageFolder
from torchvision.models import resnet101
import torch
import torchbearer

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, default=None, help='ImageNet path')
parser.add_argument('--model-file')
args = parser.parse_args()

root = os.path.join(args.dataset_path, 'train')

old_classes = os.listdir(root)
new_classes = ImageFolder(root=root).classes

model = resnet101(False)
model.load_state_dict(torch.load(args.model_file)[torchbearer.MODEL].module)

new_weights = torch.zeros_like(model.fc.weight.data)
new_bias = torch.zeros_like(model.fc.bias.data)

for layer in range(1000):
    new_layer = new_classes.index(old_classes[layer])

    new_weights[layer, :] = model.fc.weight[layer, :]
    new_bias[layer] = model.fc.bias[layer]

model.fc.weight.data = new_weights.data
model.fc.bias.data = new_weights.data

torch.save(model, args.model_file + '_converted.pt')