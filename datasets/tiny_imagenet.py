from torch.utils.data import Dataset
import os
from torchvision.datasets.folder import default_loader


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.words = self.parse_classes()

        if train:
            self.class_path = os.path.join(root, 'train')
            self.img_labels = self.parse_train()
        else:
            self.class_path = os.path.join(root, 'val')
            self.img_labels = self.parse_val_labels()

    def parse_classes(self):
        words_path = os.path.join(self.root, 'wnids.txt')
        words = {}
        i = 0
        with open(words_path, 'r') as f:
            for w in f:
                w = w.strip('\n')
                word_label = w.split('\t')[0]
                words[word_label] = i
                i += 1
        return words

    def parse_val_labels(self):
        val_annot = os.path.join(self.root, 'val', 'val_annotations.txt')
        img_label = []
        with open(val_annot, 'r') as f:
            for line in f:
                line.strip('\n')
                img, word, *_ = line.split('\t')
                img = os.path.join(self.root, 'val', 'images', img)
                img_label.append((img, self.words[word]))
        return img_label

    def parse_train(self):
        img_labels = []
        for c in os.listdir(self.class_path):
            label = self.words[c]
            images_path = os.path.join(self.root, 'train', c, 'images')
            for im in os.listdir(images_path):
                im_path = os.path.join(images_path, im)
                img_labels.append((im_path, label))
        return img_labels

    def __getitem__(self, index):
        img, label = self.img_labels[index]
        pil_img = default_loader(img)

        if self.transform is not None:
            pil_img = self.transform(pil_img)

        return pil_img, label

    def __len__(self) -> int:
        return len(self.img_labels)
