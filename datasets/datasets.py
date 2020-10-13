import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, ImageNet
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, \
    Resize, CenterCrop
import os
from datasets.tiny_imagenet import TinyImageNet
from datasets.imagenet_hdf5 import ImageNetHDF5
from utils import split, EqualSplitter, auto_augment, _fa_reduced_cifar10
from datasets.toxic import toxic_ds
from datasets.toxic_bert import toxic_bert
from datasets.bengali import BengaliConsonantDiacritic, BengaliGraphemeRoot, BengaliVowelDiacritic, BengaliGraphemeWhole
from datasets.fashion import OldFashionMNIST


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


def imagenet_a_transforms(args):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([Resize(256), CenterCrop(224), ToTensor(), normalize])

    return transform


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
    'fashion_old': fashion_transforms,
    'tinyimagenet': tinyimagenet_transforms,
    'imagenet': imagenet_transforms,
    'imagenet_hdf5': imagenet_transforms,
    'imagenet_a': imagenet_a_transforms,
    'commands': commands_transforms,
    'modelnet': modelnet_transforms,
    'bengali_r': bengali_transforms,
    'bengali_c': bengali_transforms,
    'bengali_v': bengali_transforms,
    'bengali': bengali_transforms
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
def fashion_old(args):
    data = OldFashionMNIST
    transform_train, transform_test = dstransforms[args.dataset](args)

    root = './data/fashion_old'
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


def imagenet_a(args):
    data = ImageFolder
    transform = dstransforms[args.dataset](args)
    root = args.dataset_path
    testset = data(root=f'{root}/', transform=transform)

    return None, None, testset


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


@split
def bengali(args):
    transform_train = dstransforms[args.dataset](args)

    root = '/ssd/bengali' if args.dataset_path is None else args.dataset_path

    trainset = BengaliGraphemeWhole(root=root, transform=transform_train)
    return trainset


def imdb(args):
    from torchtext import data, datasets

    TEXT = data.Field(tokenize='spacy', batch_first=True)
    LABEL = data.LabelField(dtype=torch.float)

    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL, root=args.dataset_path)

    TEXT.build_vocab(train_data, vectors='fasttext.simple.300d')
    LABEL.build_vocab(train_data)

    train_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, test_data),
        batch_size=args.batch_size,
        sort_within_batch=True,
        device=args.device)

    train_iterator.vectors = TEXT.vocab.vectors.to(args.device)
    train_iterator.ntokens = len(TEXT.vocab)
    return train_iterator, None, test_iterator


class ReverseOrder(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        res = self.dataset[i]
        # print(res)
        return res[1], torch.tensor(res[0]).long()

    def __len__(self):
        return len(self.dataset)


def yelp_2(args):
    from torchtext import datasets
    from .toxic_bert import NoBatchBucketIterator

    train_data, test_data = datasets.YelpReviewPolarity(root=args.dataset_path)
    train_data, test_data = ReverseOrder(train_data), ReverseOrder(test_data)

    train_iterator = NoBatchBucketIterator(dataset=train_data, batch_size=args.batch_size,
                                           sort_key=lambda x: x[0].size(0),
                                           device=torch.device(args.device), sort_within_batch=True)
    test_iterator = NoBatchBucketIterator(dataset=test_data, batch_size=args.batch_size,
                                          sort_key=lambda x: x[0].size(0),
                                          device=torch.device(args.device), sort_within_batch=True)

    vocab = train_data.dataset.get_vocab()
    vocab.load_vectors('fasttext.simple.300d')

    train_iterator.vectors = vocab.vectors.to(args.device)
    train_iterator.ntokens = len(vocab)
    return train_iterator, None, test_iterator


def yelp_5(args):
    from torchtext import datasets
    from .toxic_bert import NoBatchBucketIterator

    train_data, test_data = datasets.YelpReviewFull(root=args.dataset_path)
    train_data, test_data = ReverseOrder(train_data), ReverseOrder(test_data)

    train_iterator = NoBatchBucketIterator(dataset=train_data, batch_size=args.batch_size,
                                           sort_key=lambda x: x[0].size(0),
                                           device=torch.device(args.device), sort_within_batch=True)
    test_iterator = NoBatchBucketIterator(dataset=test_data, batch_size=args.batch_size,
                                          sort_key=lambda x: x[0].size(0),
                                          device=torch.device(args.device), sort_within_batch=True)

    vocab = train_data.dataset.get_vocab()
    vocab.load_vectors('fasttext.simple.300d')

    train_iterator.vectors = vocab.vectors.to(args.device)
    train_iterator.ntokens = len(vocab)
    return train_iterator, None, test_iterator


ds = {
    'cifar10': cifar,
    'cifar100': cifar,
    'fashion': fashion,
    'fashion_old': fashion_old,
    'imagenet': imagenet,
    'imagenet_hdf5': imagenet_hdf5,
    'imagenet_a': imagenet_a,
    'commands': commands,
    'tinyimagenet': tinyimagenet,
    'reduced_cifar': reduced_cifar,
    'modelnet': modelnet,
    'toxic': toxic_ds,
    'toxic_bert': toxic_bert,
    'bengali_r': bengali_r,
    'bengali_c': bengali_c,
    'bengali_v': bengali_v,
    'bengali': bengali,
    'imdb': imdb,
    'yelp_2': yelp_2,
    'yelp_5': yelp_5
}

dsmeta = {
    'cifar10': {'classes': 10, 'nc': 3, 'size': (32, 32)},
    'cifar100': {'classes': 100, 'nc': 3, 'size': (32, 32)},
    'fashion': {'classes': 10, 'nc': 1, 'size': (28, 28)},
    'fashion_old': {'classes': 10, 'nc': 1, 'size': (28, 28)},
    'imagenet': {'classes': 1000, 'nc': 3, 'size': (224, 224)},
    'imagenet_hdf5': {'classes': 1000, 'nc': 3, 'size': (224, 224)},
    'imagenet_a': {'classes': 1000, 'nc': 3, 'size': (224, 224)},
    'commands': {'classes': 12, 'nc': 1, 'size': (32, 32)},
    'tinyimagenet': {'classes': 200, 'nc': 3, 'size': (64, 64)},
    'reduced_cifar': {'classes': 10, 'nc': 3, 'size': (32, 32)},
    'modelnet': {'classes': 10, 'nc': None, 'size': None},
    'toxic': {'classes': 6, 'nc': 300, 'size': [-1]},
    'toxic_bert': {'classes': 6, 'nc': None, 'size': [-1]},
    'bengali_r': {'classes': 168, 'nc': 1, 'size': (64, 64)},
    'bengali_c': {'classes': 7, 'nc': 1, 'size': (64, 64)},
    'bengali_v': {'classes': 11, 'nc': 1, 'size': (64, 64)},
    'bengali': {'classes': (168, 11, 7), 'nc': 1, 'size': (64, 64)},
    'imdb': {'classes': 1, 'nc': 300, 'size': [-1]},
    'yelp_2': {'classes': 1, 'nc': 300, 'size': [-1]},
    'yelp_5': {'classes': 5, 'nc': 300, 'size': [-1]},
}

nlp_data = ['toxic', 'toxic_bert', 'imdb', 'yelp_2', 'yelp_5']
