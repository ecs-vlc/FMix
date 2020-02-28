<div align="center">
  
# FMix

This repository contains the __official__ implementation of the paper: Understanding and Enhancing Mixed Sample Data Augmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/understanding-and-enhancing-mixed-sample-data/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=understanding-and-enhancing-mixed-sample-data)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/understanding-and-enhancing-mixed-sample-data/image-classification-on-fashion-mnist)](https://paperswithcode.com/sota/image-classification-on-fashion-mnist?p=understanding-and-enhancing-mixed-sample-data)


<p align="center">
  <a href="">ArXiv</a> •
  <a href="">Papers With Code</a> •
  <a href="#examples">Getting Started</a> •
  <a href="#install">Loading Pre-trained Models</a>
</p>

</div>

## What is FMix?

FMix is a variant of MixUp, CutMix, etc. introduced in our paper 'Understanding and Enhancing Mixed Sample Data Augmentation'. It uses masks sampled from Fourier space to mix training examples. Here's an example:

<div align="center">
  <img src="./fmix_example.png", width="60%">
</div>

## Getting Started

The core implementation of `FMix` uses `numpy` and can be found in [`fmix.py`](./fmix.py). We provide bindings for this in [PyTorch](https://pytorch.org/) (with [Torchbearer](https://github.com/pytorchbearer/torchbearer) or [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)) and [Tensorflow](https://www.tensorflow.org/).

### Torchbearer

The `FMix` callback in [`torchbearer_implementation.py`](./implementations/torchbearer_implementation.py) can be added directly to your torchbearer code:

```
from implementations.torchbearer_implementation import FMix

fmix = FMix()
trial = Trial(model, optimiser, fmix.loss(), callbacks=[fmix])
```

See an example in [`test_torchbearer.py`](./implementations/test_torchbearer.py).

### PyTorch-Lightning

For PyTorch-Lightning, we provide a class, `FMix` in [`lightning.py`](./implementations/lightning.py) that can be used in your `LightningModule`:

```
from implementations.lightning import FMix

class CoolSystem(pl.LightningModule):
    def __init__(self):
        ...
        
        self.fmix = FMix()
    
    def training_step(self, batch, batch_nb):
        x, y = batch
        x = self.fmix(x)

        x = self.forward(x)

        loss = self.fmix.loss(x, y)
        return {'loss': loss}
```

See an example in [`test_lightning.py`](./implementations/test_lightning.py).

### Tensorflow

For Tensorflow, we provide a class, `FMix` in [`tensorflow_implementation.py`](./implementations/tensorflow_implementation.py) that can be used in your tensorflow code:

```
from implementations.tensorflow_implementation import FMix

fmix = FMix()

def loss(model, x, y, training=True):
    x = fmix(x)
    y_ = model(x, training=training)
    return tf.reduce_mean(fmix.loss(y_, y))
```

See an example in [`test_tensorflow.py`](./implementations/test_tensorflow.py).

## Loading Pre-trained Models
