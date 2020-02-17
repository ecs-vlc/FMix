from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, \
    Resize, CenterCrop
import os
from datasets.tiny_imagenet import TinyImageNet
from utils import split, EqualSplitter, auto_augment, _fa_reduced_cifar10


@auto_augment(_fa_reduced_cifar10)
def cifar_transforms(args):
    normalize = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_base = [ToTensor(), normalize]

    if args.augment:
        transform = [RandomCrop(32, padding=4), RandomHorizontalFlip()] + transform_base
    else:
        transform = transform_base

    transform_train = Compose(transform)
    transform_test = Compose(transform_base)
    return transform_train, transform_test


def fashion_transforms(args):
    normalise = Normalize((0.1307,), (0.3081,))
    base = [ToTensor(), normalise]

    if args.augment:
        transform = [RandomCrop(28, padding=4), RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose(base)
    return transform_train, transform_test


def imagenet_transforms(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [ToTensor(), normalize]

    if args.augment:
        transform = [RandomResizedCrop(224), RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose([Resize(256), CenterCrop(224)] + base)
    return transform_train, transform_test


def tinyimagenet_transforms(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    base = [ToTensor(), normalize, ]

    if args.augment:
        transform = [RandomHorizontalFlip()] + base
    else:
        transform = base

    transform_train = Compose(transform)
    transform_test = Compose([*base])
    return transform_train, transform_test


def commands_transforms(args):
    from datasets.google_commands.transforms import ChangeAmplitude, ChangeSpeedAndPitchAudio, FixAudioLength, ToTensor, LoadAudio, ToMelSpectrogram
    from datasets.google_commands.sft_transforms import ToSTFT, StretchAudioOnSTFT, TimeshiftAudioOnSTFT, FixSTFTDimension, ToMelSpectrogramFromSTFT, DeleteSTFT
    augs = Compose([LoadAudio(), ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
             TimeshiftAudioOnSTFT(), FixSTFTDimension()])

    feats = Compose([ToMelSpectrogramFromSTFT(n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    train_transform = Compose([augs, feats])
    test_transform = Compose([LoadAudio(), FixAudioLength(), ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])

    return train_transform, test_transform


dstransforms = {
    'cifar10': cifar_transforms,
    'cifar100': cifar_transforms,
    'cifar10h': cifar_transforms,
    'reduced_cifar': cifar_transforms,
    'fashion': fashion_transforms,
    'tinyimagenet': tinyimagenet_transforms,
    'imagenet': imagenet_transforms,
    'commands': commands_transforms,
}


@split
def cifar(args):
    cifar = CIFAR100 if args.dataset == 'cifar100' else CIFAR10
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = args.dataset_path if args.dataset_path is not None else './data'
    trainset = cifar(root=root, train=True, download=True, transform=transform_train)
    valset = cifar(root=root, train=False, download=True, transform=transform_test)
    return trainset, valset


def reduced_cifar(args):
    cifar = CIFAR100 if args.dataset == 'cifar100' else CIFAR10
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = args.dataset_path if args.dataset_path is not None else './data'
    trainset = cifar(root=root, train=True, download=True, transform=transform_train)

    split_ds = EqualSplitter(trainset, args.split_fraction)
    trainset, valset = split_ds.get_train_dataset(), split_ds.get_val_dataset()

    testset = cifar(root=root, train=False, download=True, transform=transform_test)
    return trainset, valset, testset


@split
def fashion(args):
    data = FashionMNIST
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = './data/fashion'
    root = args.dataset_path if args.dataset_path is not None else root
    trainset = data(root=root, train=True, download=True, transform=transform_train)
    valset = data(root=root, train=False, download=True, transform=transform_test)
    return trainset, valset


@split
def imagenet(args):
    data = ImageFolder
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = '/ssd/ILSVRC2012' if args.dataset_path is None else args.dataset_path

    trainset = data(root=root, transform=transform_train)
    testset = data(root=root, transform=transform_test)

    return trainset, testset


@split
def tinyimagenet(args):
    data = TinyImageNet
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = '/ssd/tinyimagenet' if args.dataset_path is None else args.dataset_path

    trainset = data(root=root, transform=transform_train)
    valset = data(root=root, transform=transform_test, train=False)
    return trainset, valset


def commands(args):
    from datasets.google_commands.google_commands import SpeechCommandsDataset

    transform_train, transform_test = dstransforms[args.dataset](args)

    trainset = SpeechCommandsDataset(os.path.join(args.dataset_path, 'train'), transform=transform_train)
    valset = SpeechCommandsDataset(os.path.join(args.dataset_path, 'valid'), transform=transform_test)
    testset = SpeechCommandsDataset(os.path.join(args.dataset_path, 'test'), transform=transform_test)

    return trainset, valset, testset


ds = {
    'cifar10': cifar,
    'cifar100': cifar,
    'fashion': fashion,
    'fashion_old': fashion,
    'imagenet': imagenet,
    'commands': commands,
    'tinyimagenet': tinyimagenet,
    'reduced_cifar': reduced_cifar,
}

dsmeta = {
    'cifar10': {'classes': 10, 'nc': 3, 'size': 32},
    'cifar100': {'classes': 100, 'nc': 3, 'size': 32},
    'fashion': {'classes': 10, 'nc': 1, 'size': 28},
    'fashion_old': {'classes': 10, 'nc': 1, 'size': 28},
    'imagenet': {'classes': 1000, 'nc': 3, 'size': 224},
    'commands': {'classes': 12, 'nc': 1, 'size': 32},
    'tinyimagenet': {'classes': 200, 'nc': 3, 'size': 64},
    'reduced_cifar': {'classes': 10, 'nc': 3, 'size': 32},
}
