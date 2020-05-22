from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, \
    Resize, CenterCrop
import os
from datasets.tiny_imagenet import TinyImageNet
from datasets.imagenet_hdf5 import ImageNetHDF5
from utils import split, EqualSplitter, auto_augment, _fa_reduced_cifar10
from datasets.toxic import toxic_ds
from datasets.bengali import BengaliConsonantDiacritic, BengaliGraphemeRoot, BengaliVowelDiacritic


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


def modelnet_transforms(args):
    import kaolin.transforms as tfs

    if args.msda_mode in ['mixup', 'cutmix', 'alt_mixup_fmix', 'alt_mixup_cutmix', 'alt_fmix_cutmix']:
        raise RuntimeError('Mixup and CutMix callbacks not designed for 3D classification.')

    fmix_transform = tfs.Compose([
            tfs.TriangleMeshToVoxelGrid(args.pointcloud_resolution, normalize=True),
        ])
    test_transform = tfs.Compose([
            tfs.TriangleMeshToPointCloud(num_samples=1000),
            tfs.NormalizePointCloud()
        ])

    if args.msda_mode == 'fmix':
        transform = fmix_transform
    else:
        transform = test_transform

    return transform, test_transform


def bengali_transforms(args):
    import numpy as np
    from PIL import Image

    def crop_char_image(image, threshold=5. / 255.):
        assert image.ndim == 2
        is_black = image > threshold

        is_black_vertical = np.sum(is_black, axis=0) > 0
        is_black_horizontal = np.sum(is_black, axis=1) > 0
        left = np.argmax(is_black_horizontal)
        right = np.argmax(is_black_horizontal[::-1])
        top = np.argmax(is_black_vertical)
        bottom = np.argmax(is_black_vertical[::-1])
        height, width = image.shape
        cropped_image = image[left:height - right, top:width - bottom]
        return Image.fromarray(cropped_image)

    return Compose([
        crop_char_image,
        Resize((64, 64)),
        ToTensor(),
        Normalize((0.0692,), (0.2051,))
    ])


dstransforms = {
    'cifar10': cifar_transforms,
    'cifar100': cifar_transforms,
    'cifar10h': cifar_transforms,
    'reduced_cifar': cifar_transforms,
    'fashion': fashion_transforms,
    'tinyimagenet': tinyimagenet_transforms,
    'imagenet': imagenet_transforms,
    'imagenet_hdf5': imagenet_transforms,
    'commands': commands_transforms,
    'modelnet': modelnet_transforms,
    'bengali_r': bengali_transforms,
    'bengali_c': bengali_transforms,
    'bengali_v': bengali_transforms
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

    trainset = data(root=f'{root}/train', transform=transform_train)
    testset = data(root=f'{root}/val', transform=transform_test)

    return trainset, testset


@split
def imagenet_hdf5(args):
    data = ImageNetHDF5
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = '/ssd/ILSVRC2012' if args.dataset_path is None else args.dataset_path

    trainset = data(root=f'{root}/train', transform=transform_train)
    testset = data(root=f'{root}/val', transform=transform_test)

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


@split
def modelnet(args):
    from kaolin.datasets import ModelNet

    categories = ['chair', 'monitor', 'bathtub', 'bed', 'desk', 'dresser', 'night_stand', 'sofa', 'table', 'toilet']
    transform_train, transform_test = dstransforms[args.dataset](args)

    trainset = ModelNet(args.dataset_path, categories=categories, split='train', transform=transform_train, device=args.device)
    valset = ModelNet(args.dataset_path, categories=categories, split='test', transform=transform_test, device=args.device)

    return trainset, valset


@split
def bengali_r(args):
    transform_train = dstransforms[args.dataset](args)

    root = '/ssd/bengali' if args.dataset_path is None else args.dataset_path

    trainset = BengaliGraphemeRoot(root=root, transform=transform_train)
    return trainset


@split
def bengali_c(args):
    transform_train = dstransforms[args.dataset](args)

    root = '/ssd/bengali' if args.dataset_path is None else args.dataset_path

    trainset = BengaliConsonantDiacritic(root=root, transform=transform_train)
    return trainset


@split
def bengali_v(args):
    transform_train = dstransforms[args.dataset](args)

    root = '/ssd/bengali' if args.dataset_path is None else args.dataset_path

    trainset = BengaliVowelDiacritic(root=root, transform=transform_train)
    return trainset


ds = {
    'cifar10': cifar,
    'cifar100': cifar,
    'fashion': fashion,
    'fashion_old': fashion,
    'imagenet': imagenet,
    'imagenet_hdf5': imagenet_hdf5,
    'commands': commands,
    'tinyimagenet': tinyimagenet,
    'reduced_cifar': reduced_cifar,
    'modelnet': modelnet,
    'toxic': toxic_ds,
    'bengali_r': bengali_r,
    'bengali_c': bengali_c,
    'bengali_v': bengali_v
}

dsmeta = {
    'cifar10': {'classes': 10, 'nc': 3, 'size': (32, 32)},
    'cifar100': {'classes': 100, 'nc': 3, 'size': (32, 32)},
    'fashion': {'classes': 10, 'nc': 1, 'size': (28, 28)},
    'fashion_old': {'classes': 10, 'nc': 1, 'size': (28, 28)},
    'imagenet': {'classes': 1000, 'nc': 3, 'size': (224, 224)},
    'imagenet_hdf5': {'classes': 1000, 'nc': 3, 'size': (224, 224)},
    'commands': {'classes': 12, 'nc': 1, 'size': (32, 32)},
    'tinyimagenet': {'classes': 200, 'nc': 3, 'size': (64, 64)},
    'reduced_cifar': {'classes': 10, 'nc': 3, 'size': (32, 32)},
    'modelnet': {'classes': 10, 'nc': None, 'size': None},
    'toxic': {'classes': None, 'nc': None, 'size': (-1, 1)},
    'bengali_r': {'classes': 168, 'nc': 1, 'size': (64, 64)},
    'bengali_c': {'classes': 7, 'nc': 1, 'size': (64, 64)},
    'bengali_v': {'classes': 11, 'nc': 1, 'size': (64, 64)},
}

