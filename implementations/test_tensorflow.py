from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.compat.v1 import enable_eager_execution
from tensorflow.keras import datasets, layers, models

from implementations.tensorflow_implementation import FMix

enable_eager_execution()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

fmix = FMix()


def loss(model, x, y, training=True):
    x = fmix(x)
    y_ = model(x, training=training)
    return tf.reduce_mean(fmix.loss(y_, y))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)


def train(model, train_images, train_labels):
    with tf.GradientTape() as t:
        current_loss = loss(model, train_images, train_labels)
    return current_loss, t.gradient(current_loss, model.trainable_variables)


epochs = range(100)
import tqdm
epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


for epoch in epochs:
    t = tqdm.tqdm()
    for batch in train_ds.shuffle(256).batch(128):
        x, y = batch
        x, y = tf.cast(x, 'float32'), tf.cast(y, 'int32')[:,0]
        current_loss, grads = train(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        epoch_accuracy(y, model(x, training=True))
        t.update(1)
        t.set_postfix_str('Epoch: {}. Loss: {}. Acc: {}'.format(epoch, current_loss, epoch_accuracy.result()))
    t.close()