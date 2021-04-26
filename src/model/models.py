"""
    This files defines the 

"""



import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations 

from src.model.utils import load_params



params = load_params()

RESOLUTION = params['RESOLUTION']
NOISE_DIM = params['NOISE_DIM']


def generator_model():
    """
      bruh
    """
    filters_initial = 256
    kernel_size = (4, 4, 4)
    half, fourth, eighth, sixteenth, threetwo = int(RESOLUTION / 2), int(
        RESOLUTION / 4), int(RESOLUTION / 8), int(RESOLUTION / 16), int(RESOLUTION / 32)

    model = tf.keras.Sequential()

    model.add(layers.Dense(threetwo*threetwo*threetwo*filters_initial,
                           use_bias=False, input_shape=(NOISE_DIM, )))  # Latent vector z 200 dimensional
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((threetwo, threetwo, threetwo, filters_initial)))
    assert model.output_shape == (
        None, threetwo, threetwo, threetwo, filters_initial)

    model.add(layers.Conv3DTranspose(filters_initial, kernel_size,
                                     strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (
        None, sixteenth, sixteenth, sixteenth, filters_initial)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(filters_initial / 2, kernel_size,
                                     strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (
        None, eighth, eighth, eighth, filters_initial/2)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(filters_initial / 4, kernel_size,
                                     strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (
        None, fourth, fourth, fourth, filters_initial/4)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(filters_initial / 8, kernel_size,
                                     strides=(2, 2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, half, half, half, filters_initial/8)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv3DTranspose(1, kernel_size, strides=(
        2, 2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, RESOLUTION, RESOLUTION, RESOLUTION, 1)

    return model

def discriminator_model():
    """
      Change structure to more layers later (4-5 convolutional layers seems common)


      IF WE USE WASSERSTEIN LOSS WITH GRADIENT PENALTY WE NEED TO CHANGE BATCHNORMALIZATION
      TO e.g. LAYER NORMALIZATION
    """
    filters_initial = 256
    kernel_size = (4, 4, 4)

    model = tf.keras.Sequential()
    model.add(layers.Conv3D(filters_initial / 8, kernel_size, strides=(2, 2, 2), padding='same',
                            input_shape=[RESOLUTION, RESOLUTION, RESOLUTION, 1]))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(filters_initial / 4, kernel_size,
                            strides=(2, 2, 2), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(filters_initial/2, kernel_size,
                            strides=(2, 2, 2), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(filters_initial, kernel_size,
                            strides=(2, 2, 2), padding='same'))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(1, kernel_size, strides=(1, 1, 1), padding='same'))

    # model.add(layers.Activation(activations.sigmoid))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='linear'))

    return model

