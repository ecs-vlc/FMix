from torchvision import datasets, transforms, models
import torch
from torch import optim
from implementations.lightning import FMix
from pytorch_lightning import LightningModule, Trainer, data_loader


# ######### Data
print('==> Preparing data..')
classes, cifar = 10, datasets.CIFAR10

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_base = [transforms.ToTensor(), normalize]
transform = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + transform_base

transform_train = transforms.Compose(transform)
transform_test = transforms.Compose(transform_base)
trainset = cifar(root='./data', train=True, download=True, transform=transform_train)
valset = cifar(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=8)


######### Model
print('==> Building model..')


class FMixExp(LightningModule):
    def __init__(self):
        super().__init__()
        self.net = models.resnet18(False)
        self.fmix = FMix()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        x = self.fmix(x)

        x = self.forward(x)

        loss = self.fmix.loss(x, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        x, y = batch

        x = self.forward(x)

        labels_hat = torch.argmax(x, dim=1)
        val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        val_acc = torch.tensor(val_acc)

        loss = self.fmix.loss(x, y, train=False)
        output = {
            'val_loss': loss,
            'val_acc': val_acc,
        }

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = 0
        val_acc_mean = 0
        for output in outputs:
            val_loss = output['val_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_loss = torch.mean(val_loss)
            val_loss_mean += val_loss

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_loss_mean /= len(outputs)
        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    @data_loader
    def train_dataloader(self):
        return trainloader

    @data_loader
    def val_dataloader(self):
        return valloader


######### Train
print('==> Starting training..')
trainer = Trainer(gpus=1, early_stop_callback=False, max_epochs=20, checkpoint_callback=False)
mod = FMixExp()
trainer.fit(mod)
