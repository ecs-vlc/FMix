# Code modified from https://github.com/pytorchbearer/torchbearer

import torch
import torch.nn.functional as F
from torch.distributions.beta import Beta

import torchbearer
from torchbearer.callbacks import Callback
from torchbearer import metrics as m


@m.running_mean
@m.mean
class MixupAcc(m.AdvancedMetric):
    def __init__(self):
        m.super(MixupAcc, self).__init__('mixup_acc')
        self.cat_acc = m.CategoricalAccuracy().root

    def process_train(self, *args):
        m.super(MixupAcc, self).process_train(*args)
        state = args[0]

        target1 = state[torchbearer.Y_TRUE]
        target2 = target1[state[torchbearer.MIXUP_PERMUTATION]]
        _state = args[0].copy()
        _state[torchbearer.Y_TRUE] = target1
        acc1 = self.cat_acc.process(_state)

        _state = args[0].copy()
        _state[torchbearer.Y_TRUE] = target2
        acc2 = self.cat_acc.process(_state)

        return acc1 * state[torchbearer.MIXUP_LAMBDA] + acc2 * (1 - state[torchbearer.MIXUP_LAMBDA])

    def process_validate(self, *args):
        m.super(MixupAcc, self).process_validate(*args)

        return self.cat_acc.process(*args)

    def reset(self, state):
        self.cat_acc.reset(state)


class RMixup(Callback):
    """Perform mixup on the model inputs. Requires use of :meth:`MixupInputs.loss`, otherwise lambdas can be found in
    state under :attr:`.MIXUP_LAMBDA`. Model targets will be a tuple containing the original target and permuted target.

    .. note::

        The accuracy metric for mixup is different on training to deal with the different targets,
        but for validation it is exactly the categorical accuracy, despite being called "val_mixup_acc"

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import Mixup

        # Example Trial which does Mixup regularisation
        >>> mixup = Mixup(0.9)
        >>> trial = Trial(None, criterion=Mixup.mixup_loss, callbacks=[mixup], metrics=['acc'])

    Args:
        lam (float): Mixup inputs by fraction lam. If RANDOM, choose lambda from Beta(alpha, alpha). Else, lambda=lam
        alpha (float): The alpha value to use in the beta distribution.
    """
    RANDOM = -10.0

    def __init__(self, alpha=1.0, lam=RANDOM, reformulate=False):
        super(RMixup, self).__init__()
        self.alpha = alpha
        self.lam = lam
        self.reformulate = reformulate
        self.distrib = Beta(self.alpha, self.alpha) if not reformulate else Beta(self.alpha + 1, self.alpha)

    @staticmethod
    def mixup_loss(state):
        """The standard cross entropy loss formulated for mixup (weighted combination of `F.cross_entropy`).

        Args:
            state: The current :class:`Trial` state.
        """
        input, target = state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE]

        if state[torchbearer.DATA] is torchbearer.TRAIN_DATA:
            y1, y2 = target
            return F.cross_entropy(input, y1) * state[torchbearer.MIXUP_LAMBDA] + F.cross_entropy(input, y2) * (1-state[torchbearer.MIXUP_LAMBDA])
        else:
            return F.cross_entropy(input, target)

    def on_sample(self, state):
        if self.lam is RMixup.RANDOM:
            if self.alpha > 0:
                lam = self.distrib.sample()
            else:
                lam = 1.0
        else:
            lam = self.lam

        state[torchbearer.MIXUP_LAMBDA] = lam

        state[torchbearer.MIXUP_PERMUTATION] = torch.randperm(state[torchbearer.X].size(0))
        state[torchbearer.X] = state[torchbearer.X] * state[torchbearer.MIXUP_LAMBDA] + \
                               state[torchbearer.X][state[torchbearer.MIXUP_PERMUTATION],:] \
                               * (1 - state[torchbearer.MIXUP_LAMBDA])

        if self.reformulate:
            state[torchbearer.MIXUP_LAMBDA] = 1


from torchbearer.metrics import default as d
d.__loss_map__[RMixup.mixup_loss.__name__] = MixupAcc
