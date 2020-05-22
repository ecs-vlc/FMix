import io
import os
import pickle

import h5py
from PIL import Image
from torchvision.datasets import VisionDataset


class ImageNetHDF5(VisionDataset):
    def __init__(self, root, transform=None):
        super(ImageNetHDF5, self).__init__(root, transform=transform, target_transform=None)

        self.dest = pickle.load(open(os.path.join(root, 'dest.p'), 'rb'))
        self.targets = {f[:-5]: i for i, f in enumerate(filter(lambda f: '.hdf5' in f, os.listdir(root)))}

    def __getitem__(self, index):
        dest, i = self.dest[index]
        with h5py.File(os.path.join(self.root, dest + '.hdf5'), 'r') as f:
            sample = f['data'][i]

        sample = Image.open(io.BytesIO(sample))
        sample = sample.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.targets[dest]

    def __len__(self):
        return len(self.dest)
