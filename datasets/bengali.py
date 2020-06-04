"""
Adapted from https://www.kaggle.com/corochann/bengali-seresnext-training-with-pytorch
"""
from torch.utils.data import Dataset
# import os
# from torchvision.datasets.folder import default_loader
import numpy as np
import pandas as pd
import gc


def prepare_image(root, indices=[0, 1, 2, 3]):
    # assert data_type in ['train', 'test']
    # if submission:
    #     image_df_list = [pd.read_parquet(datadir / f'{data_type}_image_data_{i}.parquet')
    #                      for i in indices]
    # else:
    image_df_list = [pd.read_feather(f'{root}/train_image_data_{i}.feather') for i in indices]

    HEIGHT = 137
    WIDTH = 236
    images = [df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH) for df in image_df_list]
    del image_df_list
    gc.collect()
    images = np.concatenate(images, axis=0)
    return images


class Bengali(Dataset):
    def __init__(self, root, targets, transform=None):
        self.transform = transform

        if isinstance(targets, list):
            self.labels = list(pd.read_csv(f'{root}/train.csv')[targets].itertuples(index=False, name=None))
        else:
            self.labels = pd.read_csv(f'{root}/train.csv')[targets]
        self.images = prepare_image(root)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = (255 - image).astype(np.float32) / 255.

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self) -> int:
        return len(self.labels)


class BengaliGraphemeWhole(Bengali):
    def __init__(self, root, transform=None):
        super().__init__(root, ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], transform=transform)


class BengaliGraphemeRoot(Bengali):
    def __init__(self, root, transform=None):
        super().__init__(root, 'grapheme_root', transform=transform)


class BengaliVowelDiacritic(Bengali):
    def __init__(self, root, transform=None):
        super().__init__(root, 'vowel_diacritic', transform=transform)


class BengaliConsonantDiacritic(Bengali):
    def __init__(self, root, transform=None):
        super().__init__(root, 'consonant_diacritic', transform=transform)
