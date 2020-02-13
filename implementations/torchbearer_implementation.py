import torch.nn.functional as F
import torchbearer
from torchbearer.callbacks import Callback
from fmix import sample_mask, FMixBase
import torch


def fmix_loss(input, y, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix

    Args:
        input: If train, mixed input. If not train, standard input
        y: Targets for first image
        index: Permutation for mixing
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """
    if train and not reformulate:
        y2 = y[index]
        return F.cross_entropy(input, y) * lam + F.cross_entropy(input, y2) * (1 - lam)
    else:
        return F.cross_entropy(input, y)


class FMix(FMixBase, Callback):
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

            fmix = FMix(...)
            trial = Trial(model, optimiser, fmix.loss(), callbacks=[fmix])
            # ...
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def on_sample(self, state):
        super().on_sample(state)
        x, y = state[torchbearer.X], state[torchbearer.Y_TRUE]
        device = state[torchbearer.DEVICE]

        x = self(x)

        # Store the results
        state[torchbearer.X] = x
        state[torchbearer.Y_TRUE] = y

        # Set mixup flags
        state[torchbearer.MIXUP_LAMBDA] = torch.tensor([self.lam], device=device) if not self.reformulate else torch.tensor([1], device=device)
        state[torchbearer.MIXUP_PERMUTATION] = self.index

    def __call__(self, x):
        lam, mask = sample_mask(self.alpha, self.decay_power, self.size, self.max_soft, self.reformulate)
        index = torch.randperm(x.size(0)).to(x.device)
        mask = torch.from_numpy(mask).float().to(x.device)

        # Mix the images
        x1 = mask * x
        x2 = (1 - mask) * x[index]
        self.index = index
        self.lam = lam
        return x1 + x2

    def loss(self):
        def _fmix_loss(state):
            y_pred = state[torchbearer.Y_PRED]
            y = state[torchbearer.Y_TRUE]
            index = state[torchbearer.MIXUP_PERMUTATION] if torchbearer.MIXUP_PERMUTATION in state else None
            lam = state[torchbearer.MIXUP_LAMBDA] if torchbearer.MIXUP_LAMBDA in state else None
            train = state[torchbearer.MODEL].training
            return fmix_loss(y_pred, y, index, lam, train, self.reformulate)

        return _fmix_loss


from torchbearer.metrics import default as d
from utils.reformulated_mixup import MixupAcc
d.__loss_map__[FMix().loss().__name__] = MixupAcc