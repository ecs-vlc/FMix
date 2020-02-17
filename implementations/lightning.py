import torch.nn.functional as F
from fmix import sample_mask, FMixBase
import torch


def fmix_loss(input, y1, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix

    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:
        y2 = y1[index]
        return F.cross_entropy(input, y1) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return F.cross_entropy(input, y1)


class FMix(FMixBase):
    r""" FMix augmentation

        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].

        Example
        -------

        .. code-block:: python

            class FMixExp(pl.LightningModule):
                def __init__(*args, **kwargs):
                    self.fmix = Fmix(...)
                    # ...

                def training_step(self, batch, batch_idx):
                    x, y = batch
                    x = self.fmix(x)

                    feature_maps = self.forward(x)
                    logits = self.classifier(feature_maps)
                    loss = self.fmix.loss(logits, y)

                    # ...
                    return loss
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        # Sample mask and generate random permutation
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # Mix the images
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam
        return x1+x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)
