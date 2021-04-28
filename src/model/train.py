import tensorflow as tf
import numpy as np

from src.models.util import load_params
from src.model.models import generator_model, discriminator_model
from src.model.losses import gradient_penalty, ws_disc_loss, ws_gen_loss

import os
import time




# CONSTANTS
params = load_params()


RESOLUTION = params['RESOLUTION']
BUFFER_SIZE = params['BUFFER_SIZE']  
BATCH_SIZE = params['BATCH_SIZE']
NOISE_DIM = params['NOISE_DIM']
RESTORE = True
NUM_DISC_UPDATES = 5
LAMBDA = params['LAMBDA']
NUM_GPUS = params['NUM_GPUS']

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
    f'src/data/{RESOLUTION}/*.npy', shuffle=True)
dataset = ds_files.map(process_path)
train_ds = dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split

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
        noise = tf.random.normal([int(BATCH_SIZE/NUM_GPUS), NOISE_DIM])
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

    noise = tf.random.normal([int(BATCH_SIZE/NUM_GPUS), NOISE_DIM])
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

