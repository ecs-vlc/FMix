import tensorflow as tf
import numpy as np
from fmix import sample_mask, FMixBase
softmax_cross_entropy_with_logits = tf.nn.softmax_cross_entropy_with_logits


def fmix_loss(input, y1, index, lam, train=True, reformulate=False):
    r"""Criterion for fmix

    Args:
        input: If train, mixed input. If not train, standard input
        y1: Targets for first image
        y2: Targets for image mixed with first image
        lam: Lambda value of mixing
        train: If true, sum cross entropy of input with y1 and y2, weighted by lam/(1-lam). If false, cross entropy loss with y1
    """

    if train and not reformulate:
        y2 = tf.gather(y1, index)
        y1, y2 = tf.transpose(tf.one_hot(y1, 10, axis=0)), tf.transpose(tf.one_hot(y2, 10, axis=0))
        return softmax_cross_entropy_with_logits(logits=input, labels=y1) * lam + softmax_cross_entropy_with_logits(logits=input, labels=y2) * (1-lam)
    else:
        y1 = tf.transpose(tf.one_hot(y1, 10, axis=0))
        return softmax_cross_entropy_with_logits(logits=input, labels=y1)


class FMix(FMixBase):
    r""" FMix augmentation
    
        Args:
            decay_power (float): Decay power for frequency decay prop 1/f**d
            alpha (float): Alpha value for beta distribution from which to sample mean of mask
            size ([int] | [int, int] | [int, int, int]): Shape of desired mask, list up to 3 dims
            max_soft (float): Softening value between 0 and 0.5 which smooths hard edges in the mask.
            reformulate (bool): If True, uses the reformulation of [1].

        Example
        ----------

        fmix = FMix(...)

        def loss(model, x, y, training=True):
            x = fmix(x)
            y_ = model(x, training=training)
            return tf.reduce_mean(fmix.loss(y_, y))
    """
    def __init__(self, decay_power=3, alpha=1, size=(32, 32), max_soft=0.0, reformulate=False):
        super().__init__(decay_power, alpha, size, max_soft, reformulate)

    def __call__(self, x):
        shape = [int(s) for s in x.shape][1:-1]
        lam, mask = sample_mask(self.alpha, self.decay_power, shape, self.max_soft, self.reformulate)
        index = np.random.permutation(int(x.shape[0]))
        index = tf.constant(index)
        mask = np.expand_dims(mask, -1)

        x1 = x * mask
        x2 = tf.gather(x, index) * (1 - mask)
        self.index = index
        self.lam = lam

        return x1 + x2

    def loss(self, y_pred, y, train=True):
        return fmix_loss(y_pred, y, self.index, self.lam, train, self.reformulate)
