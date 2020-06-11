import numpy as np
import time
import os
from torchbearer.cv_utils import DatasetValidationSplitter


def gen_folds(args, dataset, test_size):
    from sklearn.utils.validation import check_random_state
    from sklearn.model_selection._split import _validate_shuffle_split

    n_samples = len(dataset)
    n_train, n_test = _validate_shuffle_split(n_samples, test_size, None, default_test_size=0.1)
    rng = check_random_state(args.seed)

    train_folds = []
    test_folds = []

    for i in range(args.n_folds):
        # random partition
        permutation = rng.permutation(n_samples)
        ind_test = permutation[:n_test]
        ind_train = permutation[n_test:(n_test + n_train)]
        train_folds.append(ind_train)
        test_folds.append(ind_test)

    train_folds, test_folds = np.stack(train_folds), np.stack(test_folds)
    np.savez(args.fold_path, train=train_folds, test=test_folds)


def split(func):
    def splitting(args):
        try:
            trainset, testset = func(args)

            if args.fold == 'test':
                return trainset, testset, testset
        except:
            trainset = func(args)
            testset = None

        if args.run_id == 0 and not os.path.exists(args.fold_path):
            gen_folds(args, trainset, len(trainset) // args.n_folds)
        else:
            time.sleep(3)

        folds = np.load(args.fold_path)
        train_ids, val_ids = folds['train'][int(args.fold)], folds['test'][int(args.fold)]

        splitter = DatasetValidationSplitter(len(trainset), 0.1)
        splitter.train_ids, splitter.valid_ids = train_ids, val_ids

        trainset, valset = splitter.get_train_dataset(trainset), splitter.get_val_dataset(trainset)
        return trainset, valset, (testset if testset is not None else valset)

    return splitting
