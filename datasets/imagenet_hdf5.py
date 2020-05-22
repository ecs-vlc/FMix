import io
import os
import pickle

import h5py
from PIL import Image
from torchvision.datasets import VisionDataset


class ImageNetHDF5(VisionDataset):
    def __init__(self, root, cache_size=500, transform=None):
        super(ImageNetHDF5, self).__init__(root, transform=transform, target_transform=None)

        self.dest = pickle.load(open(os.path.join(root, 'dest.p'), 'rb'))
        self.cache = {}
        self.cache_size = cache_size
        self.targets = {f[:-5]: i for i, f in enumerate(filter(lambda f: '.hdf5' in f, os.listdir(root)))}

    def load(self, file):
        with h5py.File(os.path.join(self.root, file + '.hdf5'), 'r') as f:
            return f['data']

    def fill_cache(self):
        files = (f[:-5] for f in list(filter(lambda f: '.hdf5' in f, os.listdir(self.root)))[:self.cache_size])
        for file in files:
            self.cache[file] = self.load(file)

    def load_from_cache(self, file):
        if file in self.cache:
            return self.cache[file]
        return self.load(file)

    def __getitem__(self, index):
        dest, i = self.dest[index]

        sample = self.load_from_cache(dest)[i]

        sample = Image.open(io.BytesIO(sample))
        sample = sample.convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, self.targets[dest]

    def __len__(self):
        return len(self.dest)
