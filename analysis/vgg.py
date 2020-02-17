"""
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/models/vgg.html
"""
import torch
import torch.nn as nn


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class Cache:
    def __init__(self):
        super(Cache, self).__init__()
        self.outputs = {}

    def for_name(outer, layer_name):
        class Inner(nn.Module):
            def forward(inner, x):
                outer.outputs[layer_name] = x
                return x
        return Inner()

    def get_outputs(self):
        tmp = self.outputs
        self.outputs = {}
        return tmp


class VGG(nn.Module):
    def __init__(self, features, cache, return_cache=False, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.return_cache = return_cache
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.cache = cache
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),  # Normally 4096
            nn.ReLU(True),
            self.cache.for_name('c1'),
            nn.Dropout(),
            nn.Linear(2048, 2048),  # Normally 4096
            nn.ReLU(True),
            self.cache.for_name('c2'),
            nn.Dropout(),
            nn.Linear(2048, num_classes),  # Normally 4096
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        outs = self.cache.get_outputs()
        if self.return_cache:
            return x, outs
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    cache = Cache()
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif str(v) is v:
            layers += [cache.for_name(v)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers), cache


cfgs = {
    'A': [64, 'M', 'f1', 128, 'M', 'f2', 256, 'f3', 256, 'M', 'f4', 512, 'f5', 512, 'M', 'f6', 512, 'f7', 512, 'M', 'f8'],
    'B': [64, 'f1', 64, 'M', 'f2', 128, 'f3', 128, 'M', 'f4', 256, 'f5', 256, 'M', 'f6', 512, 'f7', 512, 'M', 'f8', 512, 'f9', 512, 'M', 'f10'],
    'D': [64, 'f1', 64, 'M', 'f2', 128, 'f3', 128, 'M', 'f4', 256, 'f5', 256, 'f6', 256, 'M', 'f7', 512, 'f8', 512, 'f9', 512, 'M', 'f10', 512, 'f11', 512, 'f12', 512, 'M', 'f13'],
    'E': [64, 'f1', 64, 'M', 'f2', 128, 'f3', 128, 'M', 'f4', 256, 'f5', 256, 'f6', 256, 'f7', 256, 'M', 'f8', 512, 'f9', 512, 'f10', 512, 'f11', 512, 'M', 'f12', 512, 'f13', 512, 'f14', 512, 'f15', 512, 'M', 'f16'],
}


def _vgg(cfg, batch_norm, **kwargs):
    model = VGG(*make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('A', False, **kwargs)


def vgg11_bn(**kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('A', True, **kwargs)


def vgg13(**kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('B', False, **kwargs)


def vgg13_bn(**kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('B', True, **kwargs)


def vgg16(**kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('D', False, **kwargs)


def vgg16_bn(**kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('D', True, **kwargs)


def vgg19(**kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('E', False, **kwargs)


def vgg19_bn(**kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    """
    return _vgg('E', True, **kwargs)


if __name__ == '__main__':
    from torch import optim
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    from torchbearer import Trial
    from torchbearer.callbacks import MultiStepLR, MostRecent, Mixup, CutMix
    from implementations.torchbearer_implementation import FMix

    for mode in ['baseline', 'mix', 'fmix', 'cutmix']:
        for i in range(0, 3):
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transform_base = [transforms.ToTensor(), normalize]

            transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + transform_base

            transform_train = transforms.Compose(transform)
            transform_test = transforms.Compose(transform_base)

            trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            valset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
            valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=8)

            vgg = vgg11_bn(return_cache=False)
            optimizer = optim.SGD(vgg.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

            app = []
            loss = nn.CrossEntropyLoss()

            if mode == 'mix':
                app = [Mixup()]
                loss = Mixup.mixup_loss
            if mode == 'fmix':
                app = [FMix(alpha=1)]
                loss = Mixup.mixup_loss
            if mode == 'cutmix':
                app = [CutMix(1.0, classes=10, mixup_loss=True)]
                loss = Mixup.mixup_loss

            trial = Trial(vgg, optimizer, loss, metrics=['acc', 'loss'], callbacks=app + [MostRecent(mode + '_' + str(i + 1) + '.pt'), MultiStepLR([100, 150])])
            trial.with_generators(train_generator=trainloader, val_generator=valloader).to('cuda')
            trial.run(200, verbose=1)
