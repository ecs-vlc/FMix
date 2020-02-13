dependencies = ['torch']

from torch.hub import load_state_dict_from_url


def _preact_resnet18(msda='fmix', pretrained=False, *args, **kwargs):
    from models import ResNet18
    model = ResNet18(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            'http://marc.ecs.soton.ac.uk/pytorch-models/cifar10/preact-resnet18/{}.pt'.format(msda), progress=True)
        model.load_state_dict(state)

    return model


def _pyramidnet(msda='fmix', pretrained=False, *args, **kwargs):
    from models import aa_PyramidNet
    model = aa_PyramidNet(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            'http://marc.ecs.soton.ac.uk/pytorch-models/cifar10/pyramidnet/{}.pt'.format(msda), progress=True)
        model.load_state_dict(state)

    return model


def preact_resnet18_cifar10_baseline(pretrained=False, *args, **kwargs):
    return _preact_resnet18('baseline', pretrained, *args, **kwargs)


def preact_resnet18_cifar10_fmix(pretrained=False, *args, **kwargs):
    return _preact_resnet18('fmix', pretrained, *args, **kwargs)


def preact_resnet18_cifar10_mixup(pretrained=False, *args, **kwargs):
    return _preact_resnet18('mixup', pretrained, *args, **kwargs)


def preact_resnet18_cifar10_fmixplusmixup(pretrained=False, *args, **kwargs):
    return _preact_resnet18('fmixplusmixup', pretrained, *args, **kwargs)


def pyramidnet_cifar10_baseline(pretrained=False, *args, **kwargs):
    return _pyramidnet('baseline', pretrained, *args, **kwargs)


def pyramidnet_cifar10_fmix(pretrained=False, *args, **kwargs):
    return _pyramidnet('fmix', pretrained, *args, **kwargs)


def pyramidnet_cifar10_mixup(pretrained=False, *args, **kwargs):
    return _pyramidnet('mixup', pretrained, *args, **kwargs)


from torch.hub import load
m = load('ecs-vlc/FMix:fmix', 'pyramidnet_cifar10_mixup', pretrained=True)