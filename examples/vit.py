import sys; sys.path.append("../")

import numpy as np
import tensorflow as tf

from buteo_eo.ai.tensorflow_utils import (
    SaveBestModel,
    TimingCallback,
    OverfitProtection,
)
from tensorflow.keras.backend import clip
from tensorflow.keras import Model, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Embedding,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    GlobalAveragePooling1D,
    Conv2D,
    Reshape,
)

# load
# data = np.load("/content/drive/MyDrive/data/roads_builds_bornholm.npz")
data = np.load("C:/Users/caspe/Desktop/data_for_testing/roads_builds_bornholm.npz")
rgbn = data["bands"]
# label = data["labels"]
label = data["labels"][:, :, :, 0] # roads: 0, buildings: 1
label = label[..., np.newaxis]


# shuffle
shuffle_mask = np.random.permutation(label.shape[0])
label = label[shuffle_mask]
rgbn = rgbn[shuffle_mask]

# split
split = int(label.shape[0] * 0.2)
x_train = rgbn[:-split]
y_train = label[:-split]

x_test = rgbn[-split:]
y_test = label[-split:]

# valsplit
split = int(label.shape[0] * 0.1)
x_train = rgbn[:-split]
y_train = label[:-split]

x_val = rgbn[-split:]
y_val = label[-split:]

input_shape = x_train.shape[1:]

class PatchExtractor(tf.keras.layers.Layer):
    """Extract patches from a tf.Layer. Only channel last format allowed."""

    def __init__(self, shape_x=16, shape_y=16):
        super(PatchExtractor, self).__init__()
        self.shape_x = shape_x
        self.shape_y = shape_y

    def call(self, images):
        sizes = [1, self.shape_x, self.shape_y, 1]
        strides = [1, self.shape_x, self.shape_y, 1]

        patches = tf.image.extract_patches(images=images, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")

        patch_count = (images.shape[1] // self.shape_x) * (images.shape[2] // self.shape_y)
        pixel_count = patches.shape[-1]

        patches_reshaped = tf.reshape(patches, [1, patch_count, pixel_count])

        return patches_reshaped


class LocationEncoder(tf.keras.layers.Layer):
    """Encode the location of a tf.layer"""
    def __init__(self, number_of_patches, projection_dimensions):
        super(LocationEncoder, self).__init__()
        self.number_of_patches = number_of_patches + 1
        self.projection_dimensions = projection_dimensions

        self.range = tf.range(start=0, limit=self.number_of_patches, delta=1)
        self.init = None

    def call(self):

        if self.init is None:
            self.init = Embedding(input_dim=self.number_of_patches, output_dim=self.projection_dimensions, trainable=True)(self.range)

        position_embedding = self.init

        expanded = tf.expand_dims(position_embedding, axis=0)
        return expanded


class PatchProjector(tf.keras.layers.Layer):
    """Linear projection and embedding of a tf.Layer."""
    def __init__(self, projection_dimensions):
        super(PatchProjector, self).__init__()
        self.random = lambda: tf.random_normal_initializer()(shape=(1, projection_dimensions))
        self.projection_dimensions = projection_dimensions
        self.init = None

    def call(self, patch):

        if self.init is None:
            self.init = tf.Variable(
                initial_value=self.random,
                trainable=True,
            )

        class_token = tf.tile(self.init, multiples=[1, 1])
        class_token = tf.reshape(class_token, (1, 1, self.projection_dimensions))

        projection = Dense(units=self.projection_dimensions)
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

        patch_shape = patches.shape
        number_of_patches = patch_shape[1]
        projection_dimensions = patch_shape[-1]

        location = LocationEncoder(number_of_patches, projection_dimensions)()
        projection = PatchProjector(projection_dimensions)(patches)

        merged = location + projection

        if self.channel_last:
            merged_shape = tf.shape(merged)
            merged = tf.reshape(
                merged,
                (merged_shape[1], merged_shape[2], merged_shape[0]),
            )

        return merged

class MLP(tf.keras.layers.Layer):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.dense1 = Dense(hidden_features, activation=tf.nn.gelu)
        self.dense2 = Dense(out_features)
        self.dropout = Dropout(dropout_rate)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout(x)
        x = self.dense2(x)
        y = self.dropout(x)
        return y

class Block(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=4, dropout_rate=0.1):
        super(Block, self).__init__()
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MLP(projection_dim * 2, projection_dim, dropout_rate)

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x) # encoded_patches
        # Create a multi-head attention layer.
        attention_output = self.attn(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, x]) #encoded_patches
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        y = Add()([x3, x2])
        return y


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_heads=4, num_blocks=12, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.blocks = [Block(projection_dim, num_heads=self.num_heads, dropout_rate=self.dropout_rate) for _ in range(self.num_blocks)]
        self.norm = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(0.5)

    def call(self, x):
        # Create a [batch_size, projection_dim] tensor.
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        y = self.dropout(x)
        return y

class PatchExtractor(tf.keras.layers.Layer):
    """Extract patches from a tf.Layer. Only channel last format allowed."""

    def __init__(self, patch_shape=(16, 16)):
        super(PatchExtractor, self).__init__()
        self.shape_x, self.shape_y = patch_shape
        self.processed = None

    def call(self, image):
        if self.processed is None:
            patch_size = (self.shape_x, self.shape_y)

            sizes = [1, patch_size[0], patch_size[1], 1]
            strides = [1, patch_size[0], patch_size[1], 1]

            patches = tf.image.extract_patches(images=image, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")

            patch_count = (image.shape[1] // patch_size[0]) * (image.shape[2] // patch_size[1])
            pixel_count = patches.shape[-1]

            self.processed = tf.reshape(patches, [tf.shape(image)[0], patch_count, patch_size[0], patch_size[1], image.shape[-1]])

        return self.processed


class PatchReshaper(tf.keras.layers.Layer):
    """Extract patches from a tf.Layer. Only channel last format allowed."""

    def __init__(self, patch_shape=(128, 128)):
        super(PatchReshaper, self).__init__()
        self.shape_x, self.shape_y = patch_shape
        self.processed = None

    def call(self, image):
        if self.processed is None:
            self.processed = tf.reshape(image, [tf.shape(image)[0], self.shape_x, self.shape_y, image.shape[-1]])
        
        return self.processed 


class DenseApplier(tf.keras.layers.Layer):
    """Extract patches from a tf.Layer. Only channel last format allowed."""
    def __init__(self, patch_shape=(128, 128)):
        super(DenseApplier, self).__init__()
        self.processed = None

    def call(self, image):
        if self.processed is None:
            patches = PatchExtractor()(image)
            dense_layer = Dense(1024, activation="relu")(patches)
            self.processed  = PatchReshaper()(dense_layer)

        return self.processed

def create_model(num_classes, num_heads=4, num_blocks=12, dropout_rate=0.1):
    model_input = Input(shape=input_shape)

    patches = VitEncoder()(model_input)

    projection_dim = patches.shape[-1]

    representation = TransformerEncoder(projection_dim, num_heads=num_heads, num_blocks=num_blocks, dropout_rate=dropout_rate)(patches)    
    representation = GlobalAveragePooling1D()(representation)

    output_shape = (input_shape[0], input_shape[1], num_classes)
    output_pixels = input_shape[0] * input_shape[1] * num_classes

    ready_out = Dense(output_pixels, activation=tf.nn.gelu)(representation)

    ready_out_shaped = Reshape(output_shape)(ready_out)

    output = Conv2D(
        num_classes,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_initializer="glorot_normal",
        name=f"output",
        dtype="float32",
    )(ready_out_shaped)
    
    return Model(
        inputs=model_input,
        outputs=clip(output, min_value=0, max_value=100),
    )

model = create_model(1, num_heads=2, num_blocks=4)
model.summary()

shape = x_train.shape[1:]

epochs_per_fit = 10
fits = [
    { "epochs": epochs_per_fit, "bs": 1, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 2, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 4, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 8, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 16, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 32, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 64, "lr": 0.0001 },
    { "epochs": epochs_per_fit, "bs": 128, "lr": 0.0001 },
]

cur_sum = 0
for idx, val in enumerate(fits):
    fits[idx]["ie"] = cur_sum
    cur_sum += fits[idx]["epochs"]

min_delta = 0.005
model_name = "basic_roads"
monitor = "val_loss"

optimizer = tf.keras.optimizers.Adam(
    learning_rate=fits[0]["lr"],
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

model.compile(
    optimizer=optimizer,
    loss="mse",
    metrics=["mse", "mae"], 
)

print("Test dataset:")
model.evaluate(x=x_test, y=y_test, batch_size=64)

print("Validation test:")
val_loss, _mse, _mae = model.evaluate(x=x_val, y=y_val, batch_size=64)

# This ensures that the weights of the best performing model is saved at the end
save_best_model = SaveBestModel(save_best_metric=monitor, initial_weights=model.get_weights())

# Reduces the amount of total epochs by early stopping a new fit if it is not better than the previous fit.
best_val_loss = val_loss

for phase in range(len(fits)):
    use_epoch = fits[phase]["epochs"]
    use_bs = fits[phase]["bs"]
    use_lr = fits[phase]["lr"]
    use_ie = fits[phase]["ie"]

    model.optimizer.lr.assign(use_lr)
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        shuffle=True,
        epochs=use_epoch + use_ie,
        initial_epoch=use_ie,
        verbose=1,
        batch_size=use_bs,
        use_multiprocessing=True,
        workers=0,
        callbacks=[
            save_best_model,
            EarlyStopping(          # it gives the model 3 epochs to improve results based on val_loss value, if it doesnt improve-drops too much, the model running
                monitor=monitor,    # is stopped. If this continues, it would be overfitting (refer to notes)
                patience=5 if phase != 0 else 10,
                min_delta=min_delta,
                mode="min",         # loss is suppose to minimize
                baseline=best_val_loss,     # Fit has to improve upon baseline
                restore_best_weights=True,  # If stopped early, restore best weights.
            ),
            OverfitProtection(
                patience=3, # 
                difference=0.2, # 20% overfit allowed
                offset_start=1, # disregard overfit for the first epoch
            ),
            TimingCallback(
                monitor=[
                    # "loss", "val_loss",
                    "mse", "val_mse",
                    "mae", "val_mae",
                ],
            ),
        ],
    )

    # Saves the val loss to the best_val_loss for early stopping between fits.
    model.set_weights(save_best_model.best_weights)
    
    print("Validation test:")
    val_loss, _mse, _mae = model.evaluate(x=x_val, y=y_val, batch_size=256) # it evaluates the accuracy of the model we just created here
    best_val_loss = val_loss

    print("Test dataset:")
    model.evaluate(x=x_test, y=y_test, batch_size=256)

print("Finished...")