"""Implementing a ResNet-34.
Helped by the book:
"Hands-on Machine Learning with scikit-learn, keras and tensorflow"
by Aurelien Geron.

Here's our deep structure:

- Conv2d(64) * 3
- Conv2d(128) * 4
- Conv2d(256) * 6
- Conv2d(512) * 3
"""
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.layers import BatchNormalization


class ResidualUnit(keras.layers.Layer): 
    def __init__(self, filters, strides=1, activation='relu'):
        super().__init__()
        self.activation = keras.activations.get(activation)

        # Residual Unit layers
        self.layers = [
            Conv2D(filters, 3, strides=strides, padding='same',
                   use_bias=False),
            BatchNormalization(),
            self.activation,
            Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
            BatchNormalization()
        ]

        # Skip layers
        self.skip_layers = []
        # If there is a change in shape we use a conv layer to adapt properly
        if strides > 1:
            self.skip_layers = [
                Conv2D(filters, 1, strides=strides, padding='same'),
                BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.layers:
            Z = layer(Z)

        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


model = Sequential()
# Input Layers
model.add(Conv2D(64, 7, strides=2, input_shape=[256, 256, 3],
                 padding='same'))
model.add(BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(MaxPool2D(pool_size=3, strides=2, padding='same'))

# Deep Layers
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))

# Output Dense
model.add(keras.layers.GlobalAvgPool2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))