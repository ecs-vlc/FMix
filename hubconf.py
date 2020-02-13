dependencies = ['torch']

from torch.hub import load_state_dict_from_url


def preact_resnet18_fmix(pretrained=False, *args, **kwargs):
    from models import ResNet18
    model = ResNet18(*args, **kwargs)

    if pretrained:
        state = load_state_dict_from_url(
            'http://marc.ecs.soton.ac.uk/pytorch-models/preact-resnet18/fmix.pt', progress=True)
        model.load_state_dict(state)

    return model
