from torchvision.datasets import MNIST


class OldFashionMNIST(MNIST):
    """The original version of the `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset. This had
    some train / test duplicates and so was replaced. Do not use in practice, may be helpful for reproducing results
    from FMix, RandomErase and other papers.

    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
            and  ``Fashion-MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://github.com/zalandoresearch/fashion-mnist/raw/949006594cc2f804a93b9155849734564c3545ec/data/fashion/train-images-idx3-ubyte.gz',
        'https://github.com/zalandoresearch/fashion-mnist/raw/949006594cc2f804a93b9155849734564c3545ec/data/fashion/train-labels-idx1-ubyte.gz',
        'https://github.com/zalandoresearch/fashion-mnist/raw/949006594cc2f804a93b9155849734564c3545ec/data/fashion/t10k-images-idx3-ubyte.gz',
        'https://github.com/zalandoresearch/fashion-mnist/raw/949006594cc2f804a93b9155849734564c3545ec/data/fashion/t10k-labels-idx1-ubyte.gz',
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
