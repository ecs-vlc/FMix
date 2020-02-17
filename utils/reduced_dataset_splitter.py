import random
from collections import defaultdict

from torchbearer.cv_utils import SubsetDataset


class EqualSplitter:
    """ Splits a dataset into two parts, taking equal number of samples from each class

    :param dataset:
    :param split_fraction:
    """
    def __init__(self, dataset, split_fraction):
        self.ds = dataset
        self.split_fraction = split_fraction
        self.train_ids, self.valid_ids = self._gen_split_ids()

    def _gen_split_ids(self):
        classes = defaultdict(list)
        for i in range(len(self.ds)):
            _, label = self.ds[i]
            classes[label].append(i)

        nc = len(classes.keys())
        cut_per_class = int(len(self.ds) * self.split_fraction / nc)

        cut_indexes = []
        retained_indexes = []
        for c in classes.keys():
            random.shuffle(classes[c])
            cut_indexes += classes[c][:cut_per_class]
            retained_indexes += classes[c][cut_per_class:]
        return cut_indexes, retained_indexes

    def get_train_dataset(self):
        """ Creates a training dataset from existing dataset

        Args:
            dataset (torch.utils.data.Dataset): Dataset to be split into a training dataset

        Returns:
            torch.utils.data.Dataset: Training dataset split from whole dataset
        """
        return SubsetDataset(self.ds, self.train_ids)

    def get_val_dataset(self):
        """ Creates a validation dataset from existing dataset

        Args:
        dataset (torch.utils.data.Dataset): Dataset to be split into a validation dataset

        Returns:
            torch.utils.data.Dataset: Validation dataset split from whole dataset
        """
        return SubsetDataset(self.ds, self.valid_ids)

