"""
This module provides utility functions for working with Tensorflow. Additional loss functions and overfit checks.

TODO:
    - Improve documentation
"""

from datetime import datetime

import tensorflow as tf
from tensorflow.keras.layers import Activation, Embedding, Dense
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error


def tpe(y_true, y_pred):
    epsilon = 1e-7
    pred_sum = tf.math.reduce_sum(y_pred)
    true_sum = tf.math.reduce_sum(y_true)
    ratio = tf.math.divide(pred_sum, true_sum + epsilon)

    return ratio


def tpe_target(y_true, y_pred):
    epsilon = 1e-7
    pred_sum = tf.math.reduce_sum(y_pred)
    true_sum = tf.math.reduce_sum(y_true)
    ratio = tf.math.divide(pred_sum, true_sum + epsilon)

    return tf.math.abs(1 - ratio)


def mse_mae_mix_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return tf.math.multiply(mse, mae)


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


def load_mish():
    get_custom_objects().update({"Mish": Mish(mish)})


class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"

# From user2646087 @ GitHub
def timedelta_format(td_object):
    seconds = int(td_object.total_seconds())
    periods = [
        ('d', 60*60*24),
        ('h', 60*60),
        ('m', 60),
        ('s', 1)
    ]

    strings=[]
    for period_name, period_seconds in periods:
        if seconds > period_seconds:
            period_value , seconds = divmod(seconds, period_seconds)
            strings.append(f"{period_value}{period_name}")

    return " ".join(strings)


class PatchExtractor(tf.keras.layers.Layer):
    """Extract patches from a tf.Layer. Only channel last format allowed."""

    def __init__(self, shape_x=16, shape_y=16):
        super(PatchExtractor, self).__init__()
        self.shape_x = shape_x
        self.shape_y = shape_y

    def call(self, images):
        if len(tf.shape(images)) == 3:
            images = tf.expand_dims(images, axis=0)

        sizes = [1, self.shape_x, self.shape_y, 1]
        strides = [1, self.shape_x, self.shape_y, 1]

        patches = tf.image.extract_patches(
            images=images,
            sizes=sizes,
            strides=strides,
            rates=[1, 1, 1, 1],
            padding="VALID",
        )

        batch_size = tf.shape(images)[0]
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])

        return patches

class LocationEncoder(tf.keras.layers.Layer):
    """Encode the location of a tf.layer"""

    def call(self, patch):
        patch_shape = tf.shape(patch)
        number_of_patches = patch_shape[1] + tf.constant(1, tf.uint8)
        projection_dimensions = patch_shape[-1]

        positions = tf.range(start=tf.constant(0, tf.int8), limit=number_of_patches, delta=1)
        position_embedding = Embedding(
            input_dim=projection_dimensions,
            output_dim=projection_dimensions,
            trainable=True,
        )(positions)

        return tf.expand_dims(position_embedding, axis=0)

class PatchProjector(tf.keras.layers.Layer):
    """Linear projection and embedding of a tf.Layer."""

    def call(self, patch):
        patch_shape = tf.shape(patch)
        batch = tf.shape(patch)[0]
        projection_dimensions = patch_shape[-1]

        w_init = tf.random_normal_initializer()
        class_token = w_init(shape=(1, projection_dimensions))
        class_token = tf.Variable(initial_value=class_token, trainable=True)
        class_token = tf.tile(class_token, multiples = [batch, 1])
        class_token = tf.reshape(class_token, (batch, 1, projection_dimensions))

        projection = Dense(units=projection_dimensions)
        patches_embed = projection(patch)
        patches_embed = tf.concat([patches_embed, class_token], 1)

        return patches_embed

class VitEncoder(tf.keras.layers.Layer):
    def __init__(self, shape_x=16, shape_y=16, channel_last=False):
        super(VitEncoder, self).__init__()
        self.shape_x = shape_x
        self.shape_y = shape_y
        self.channel_last = channel_last

    def call(self, image):
        patches = PatchExtractor(shape_x=self.shape_x, shape_y=self.shape_y)(image)
        print(patches.shape)
        location = LocationEncoder()(patches)
        print(location.shape)
        projection = PatchProjector()(patches)
        print(projection.shape)

        merged = location + projection

        if self.channel_last:
            merged_shape = tf.shape(merged)
            merged = tf.reshape(
                merged,
                (merged_shape[1], merged_shape[2], merged_shape[0]),
            )

        return merged



class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric="val_loss", this_max=False, initial_weights=None):
        self.save_best_metric = save_best_metric
        self.max = this_max

        if initial_weights is not None:
            self.best_weights = initial_weights

        if this_max:
            self.best = float("-inf")
        else:
            self.best = float("inf")

    def on_epoch_end(self, _epoch, logs=None):
        metric_value = abs(logs[self.save_best_metric])
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, monitor=["loss", "val_loss"]):
        self.time_started = None
        self.time_finished = None
        self.monitor = monitor
        
    def on_train_begin(self, logs=None):
        self.time_started = datetime.now()
        print(f'\nTraining started: {self.time_started.strftime("%Y-%m-%d %H:%M:%S")}\n')
        
    def on_train_end(self, logs=None):
        self.time_finished = datetime.now()
        train_duration = (self.time_finished - self.time_started)
        print(f'\nTraining finished: {self.time_finished.strftime("%Y-%m-%d %H:%M:%S")}, duration: {timedelta_format(train_duration)}')
        
        metrics = [] 
        for metric in self.monitor:
            str_val = str(logs[metric])
            before_dot = len(str_val.split(".")[0])

            spaces = 16 - (len(metric) + before_dot)
            if spaces <= 0:
                spaces = 1

            pstr = f"{metric}:{' ' * spaces}{logs[metric]:.4f}"
            metrics.append(pstr)

        print('\n'.join(metrics))


class OverfitProtection(tf.keras.callbacks.Callback):
    def __init__(self, difference=0.1, patience=3, offset_start=3, verbose=True):
        self.difference = difference
        self.patience = patience
        self.offset_start = offset_start
        self.verbose = verbose
        self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        loss = logs['loss']
        val_loss = logs['val_loss']
        
        if epoch < self.offset_start:
            return

        epsilon = 1e-7
        ratio = loss / (val_loss + epsilon)

        if (1.0 - ratio) > self.difference:
            self.count += 1

            if self.verbose:
                print(f"Overfitting.. Patience: {self.count}/{self.patience}")

        elif self.count != 0:
            self.count -= 1
        
        if self.count >= self.patience:
            self.model.stop_training = True

            if self.verbose:
                print(f"Training stopped to prevent overfitting. Difference: {ratio}, Patience: {self.count}/{self.patience}")


class LearningRateAdjuster(tf.keras.callbacks.Callback):
    def __init__(self, start_epoch, decay_rate=0.95, decay_rate_epoch=10, set_at_end="end_rate", step_wise=True, verbose=True):
        self.start_epoch = start_epoch
        self.decay_rate = decay_rate
        self.decay_rate_epoch = decay_rate_epoch
        self.set_at_end = set_at_end
        self.step_wise = step_wise
        self.initial_epoch = 0
        self.decay_count = 0
        self.verbose = verbose
        self.initial_lr = None

    def on_epoch_end(self, epoch, logs=None):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * self.decay_rate ^ ((epoch - self.start_epoch) / self.decay_rate_epoch)

        if self.step_wise and epoch == (self.start_epoch + (self.decay_rate_epoch * (self.decay_count + 1))):
            if self.verbose:
                print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

            self.model.optimizer.lr.assign(new_lr)
            self.decay_count += 1

        elif not self.step_wise:
            if self.verbose:
                print(f"\nEpoch: {epoch}. Reducing Learning Rate from {old_lr} to {new_lr}")

            self.model.optimizer.lr.assign(new_lr)
    
    def on_epoch_begin(self, epoch, logs=None):
        if self.initial_lr is None:
            self.initial_lr = self.model.optimizer.lr.read_value()

    def on_train_end(self, logs=None):
        if self.set_at_end == "end_rate":
            pass
        elif isinstance(self.set_at_end, (int, float)):
            self.model.optimizer.lr.assign(self.set_at_end)
        elif self.set_at_end == "initial":
            self.model.optimizer.lr.assign(self.initial_lr)
