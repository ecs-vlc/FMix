from torchvision import datasets, transforms, models
import torch
from torch import optim
from implementations.torchbearer_implementation import FMix
from torchbearer import Trial


# ######### Data
print('==> Preparing data..')
classes, cifar = 10, datasets.CIFAR10

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_base = [transforms.ToTensor(), normalize]
transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + transform_base

transform_train = transforms.Compose(transform)
transform_test = transforms.Compose(transform_base)
trainset = cifar(root='./data', train=True, download=True, transform=transform_train)
valset = cifar(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=8)


######### Model
print('==> Building model..')
net = models.resnet18(False)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
fmix = FMix()
criterion = fmix.loss()


######### Trial
print('==> Starting training..')
trial = Trial(net, optimizer, criterion, metrics=['acc', 'loss'], callbacks=[fmix])
trial.with_generators(train_generator=trainloader, val_generator=valloader).to('cuda')
trial.run(100, verbose=2)
