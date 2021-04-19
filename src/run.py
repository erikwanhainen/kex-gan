import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations

import os
import time
#

# CONSTANTS
RESOLUTION = 128
BUFFER_SIZE = 30  # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BATCH_SIZE = 32
NOISE_DIM = 200
RESTORE = False
NUM_DISC_UPDATES = 5
LAMBDA = 10

##### CREATE DATASET FROM NPY FILES ##########


def read_npy_file(filename):
    data = np.load(filename.numpy().decode()).reshape(
        RESOLUTION, RESOLUTION, RESOLUTION, 1)
    return data.astype(np.float32)


def process_path(file_path):
    """
      Read npy file
    """
    image = tf.py_function(read_npy_file, [file_path], [tf.float32, ])
    return image


#########################################
# Define generator and discriminator models


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


def gradient_penalty(real_data, fake_data, disc):
    """
      real_data: shapes from batch
      fake_data: generated samples
    """
    bs = int(BATCH_SIZE/4)
    alpha = tf.random.uniform(shape=[bs, 1, 1, 1, 1], minval=0., maxval=1.)
    difference = fake_data - real_data
    interpolates = real_data + (alpha*difference)
#    inter = []
#    for i in range(bs):
#        inter.append(difference[i] * alpha[i])
#    inter = tf.stack(inter)
#    interpolates = real_data + inter

    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        prediction = disc(interpolates)
    gradients = tape.gradient(prediction, interpolates)
    # Used by https://github.com/timsainb/tensorflow2-generative-models/blob/master/3.0-WGAN-GP-fashion-mnist.ipynb
    # slopes = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
    # Used by 3D-IWGAN & improved_wgan
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

    slopes = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2, 3, 4]))  # Try this
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    return gradient_penalty


def ws_disc_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def ws_gen_loss(fake_output):
    return -tf.reduce_mean(fake_output)


##### Save and restore model during training #####
checkpoint_dir = './src/checkpoints/w_128'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

##### DEFINE STRATEGY AND INIT MODELS, OPTIMIZERS #####

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create the models
    generator = generator_model()
    discriminator = discriminator_model()

    ## Define optimizer  ##
    # parameters taken from https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py
    generator_optimizer = tf.keras.optimizers.Adam(
        1e-4, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = tf.keras.optimizers.Adam(
        1e-4, beta_1=0.5, beta_2=0.9)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

##### Create dataset ####
ds_files = tf.data.Dataset.list_files(
    f'../data/{RESOLUTION}/*.npy', shuffle=True)
dataset = ds_files.map(process_path)
train_ds = dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split
# Create distributed dataset depending on strategy
dist_ds = strategy.experimental_distribute_dataset(train_ds)


# TRAINING LOOP
EPOCHS = 60


def train(dataset, epochs, disc_updates):
    count = 0
    for e in range(epochs):
        for image_batch in dist_ds:
            dist_train_step(image_batch, disc_updates)
            count += 1
            print(count*BATCH_SIZE, flush=True)
        checkpoint.save(file_prefix=checkpoint_prefix)
        count = 0
        print('----- EPOCH DONE -----', e)


@tf.function
def dist_train_step(dataset_input, disc_updates):
    strategy.run(train_step, args=(dataset_input, disc_updates))


def train_step(images, disc_updates):
    """
      NOTE: we can change how often the generator is updated vs discriminator
    """

    for _ in range(disc_updates):
        noise = tf.random.normal([int(BATCH_SIZE/4), NOISE_DIM])
        with tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            disc_ws_loss = ws_disc_loss(real_output, fake_output)
            g_pen = gradient_penalty(
                images[0], generated_images, discriminator)
            disc_loss = disc_ws_loss + LAMBDA*g_pen

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables))

    noise = tf.random.normal([int(BATCH_SIZE/4), NOISE_DIM])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(
            generated_images, training=True)

        gen_loss = ws_gen_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))


print("Starting train")
if RESTORE:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('Restoring last model')

train(dist_ds, EPOCHS, NUM_DISC_UPDATES)
