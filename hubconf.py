dependencies = ['torch', 'torchvision']

from torch.hub import load_state_dict_from_url


def _preact_resnet18(msda='fmix', pretrained=False, repeat=0, *args, **kwargs):
    from models import ResNet18
    model = ResNet18(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            f'http://marc.ecs.soton.ac.uk/pytorch-models/cifar10/preact-resnet18/cifar10_preact_resnet18_{msda}_{repeat}.pt', progress=True)
        model.load_state_dict(state)

    return model


def _resnet101(msda='fmix', pretrained=False, *args, **kwargs):
    from torchvision.models.resnet import resnet101
    model = resnet101(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            f'http://marc.ecs.soton.ac.uk/pytorch-models/imagenet/resnet101/imagenet_resnet101_{msda}.pt', progress=True)
        model.load_state_dict(state)

    return model


def _pyramidnet(msda='fmix', pretrained=False, *args, **kwargs):
    from models import aa_PyramidNet
    model = aa_PyramidNet(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            f'http://marc.ecs.soton.ac.uk/pytorch-models/cifar10/pyramidnet/cifar10_pyramidnet_{msda}.pt', progress=True)
        model.load_state_dict(state)

    return model


def preact_resnet18_cifar10_baseline(pretrained=False, repeat=0, *args, **kwargs):
    return _preact_resnet18('baseline', pretrained, repeat, *args, **kwargs)


def preact_resnet18_cifar10_fmix(pretrained=False, repeat=0, *args, **kwargs):
    return _preact_resnet18('fmix', pretrained, repeat, *args, **kwargs)


def preact_resnet18_cifar10_mixup(pretrained=False, repeat=0, *args, **kwargs):
    return _preact_resnet18('mixup', pretrained, repeat, *args, **kwargs)


def preact_resnet18_cifar10_cutmix(pretrained=False, repeat=0, *args, **kwargs):
    return _preact_resnet18('cutmix', pretrained, repeat, *args, **kwargs)


def preact_resnet18_cifar10_fmixplusmixup(pretrained=False, repeat=0, *args, **kwargs):
    return _preact_resnet18('fmixplusmixup', pretrained, repeat, *args, **kwargs)


def pyramidnet_cifar10_baseline(pretrained=False, *args, **kwargs):
    return _pyramidnet('baseline', pretrained, *args, **kwargs)


def pyramidnet_cifar10_fmix(pretrained=False, *args, **kwargs):
    return _pyramidnet('fmix', pretrained, *args, **kwargs)


def pyramidnet_cifar10_mixup(pretrained=False, *args, **kwargs):
    return _pyramidnet('mixup', pretrained, *args, **kwargs)


def pyramidnet_cifar10_cutmix(pretrained=False, *args, **kwargs):
    return _pyramidnet('cutmix', pretrained, *args, **kwargs)


def renset101_imagenet_baseline(pretrained=False, *args, **kwargs):
    return _resnet101('baseline', pretrained, *args, **kwargs)


def renset101_imagenet_fmix(pretrained=False, *args, **kwargs):
    return _resnet101('fmix', pretrained, *args, **kwargs)


def renset101_imagenet_mixup(pretrained=False, *args, **kwargs):
    return _resnet101('mixup', pretrained, *args, **kwargs)


def renset101_imagenet_cutmix(pretrained=False, *args, **kwargs):
    return _resnet101('cutmix', pretrained, *args, **kwargs)
