dependencies = ['torch']

from torch.hub import load_state_dict_from_url


def _preact_resnet18(msda='fmix', pretrained=False, *args, **kwargs):
    from models import ResNet18
    model = ResNet18(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            'http://marc.ecs.soton.ac.uk/pytorch-models/preact-resnet18/{}.pt'.format(msda), progress=True)
        model.load_state_dict(state)

    return model


def preact_resnet18_baseline(pretrained=False, *args, **kwargs):
    return _preact_resnet18('baseline', pretrained, *args, **kwargs)


def preact_resnet18_fmix(pretrained=False, *args, **kwargs):
    return _preact_resnet18('fmix', pretrained, *args, **kwargs)


def preact_resnet18_mixup(pretrained=False, *args, **kwargs):
    return _preact_resnet18('mixup', pretrained, *args, **kwargs)


def preact_resnet18_fmixplusmixup(pretrained=False, *args, **kwargs):
    return _preact_resnet18('fmixplusmixup', pretrained, *args, **kwargs)
