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
BATCH_SIZE = 16
NOISE_DIM = 200
RESTORE = True
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
    bs = int(BATCH_SIZE)
    alpha = tf.random.uniform(shape=[bs, 1], minval=0., maxval=1.)
    difference = fake_data - real_data
    inter = []
    for i in range(bs):
        inter.append(difference[i] * alpha[i])
    inter = tf.stack(inter)
    interpolates = real_data + inter

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
checkpoint_dir = './src/checkpoints/w_128_v2/w_128'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

##### DEFINE STRATEGY AND INIT MODELS, OPTIMIZERS #####

#strategy = tf.distribute.MirroredStrategy()

#with strategy.scope():
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

ds_files_val = tf.data.Dataset.list_files(
    f'src/data/{RESOLUTION}_val/*.npy', shuffle=True)
dataset_val = ds_files_val.map(process_path)
val_ds = dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split
# Create distributed dataset depending on strategy
#dist_ds = strategy.experimental_distribute_dataset(train_ds)


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
        noise = tf.random.normal([int(BATCH_SIZE/2), NOISE_DIM])
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

    noise = tf.random.normal([int(BATCH_SIZE/2), NOISE_DIM])
    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)

        fake_output = discriminator(
            generated_images, training=True)

        gen_loss = ws_gen_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))

def plot(image, cut_off):
    image = image > cut_off
    figure = plt.figure()
    ax = figure.add_subplot(111, projection ='3d')
    z, x, y = image.nonzero()
    ax.scatter(x, y, z)
    ax.set_xlim3d(0, RESOLUTION)
    ax.set_ylim3d(0, RESOLUTION)
    ax.set_zlim3d(0, RESOLUTION)
    plt.show()

print("Starting train")
if RESTORE:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print('Restoring last model')

#train(dist_ds, EPOCHS, NUM_DISC_UPDATES)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3
from scipy.interpolate import interp1d


def plot_images(columns=5, rows=5, cutoff=0.4):
    '''
    Function plotting columns * rows images from CIFAR-10.
    '''
    noise = None
    while True:
        i = input()
        if i == 'exit':
            break
        if i == 'save 1':
            with open('1.npy', 'wb') as f:
                np.save(f, noise.numpy())
            continue
        if i == 'save 2':
            with open('2.npy', 'wb') as f:
                np.save(f, noise.numpy())
            continue
        if i == 'gif':
            animated_plot(amount=50, cutoff=0.5)
            continue
        fig=plt.figure()
        print('ploting')
        for i in range(columns*rows):
            noise = tf.random.normal([1, NOISE_DIM])
            generated_image = generator(noise, training=False)
            img = generated_image[0, :, :, :, 0].numpy() # IMG IS float32 type !
            img = img > cutoff
            z, x, y = img.nonzero()
            ax = fig.add_subplot(rows,columns,i+1, projection='3d')
            ax.set_xlim3d(0, RESOLUTION)
            ax.set_ylim3d(0, RESOLUTION)
            ax.set_zlim3d(0, RESOLUTION)
            ax.scatter(x, y, z, s=1)
        plt.show()


def animated_plot(amount=1, save=False, cutoff=0.4):
    def update_plot(num):
        generated_image = generator(noise[num], training=False)
        img = generated_image[0, :, :, :, 0].numpy()
        img = img > cutoff
        z, x, y = img.nonzero()
        plot._offsets3d = (x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    noise = [np.random.random([1, NOISE_DIM]) for _ in range(amount)]
#    fst = np.random.random(NOISE_DIM)
#    snd = np.random.random(NOISE_DIM)
   # if two saved noise
    fst_np = np.load('1.npy')
    fst = tf.convert_to_tensor(fst_np, dtype=tf.float32)
    snd_np = np.load('2.npy')
    snd = tf.convert_to_tensor(snd_np, dtype=tf.float32)
    linfit = interp1d([1,amount], np.vstack([fst, snd]), axis=0)
    n = [i for i in range(1, amount+1)]
    noise = []
    arr = linfit(n)
    for i in arr:
        noise.append(np.reshape(i, (1, NOISE_DIM)))
    data = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))
    z, x, y = data.nonzero()
    plot = ax.scatter(x, y, z, s=1)
    ax.set_xlim3d(0, RESOLUTION)
    ax.set_ylim3d(0, RESOLUTION)
    ax.set_zlim3d(0, RESOLUTION)
    anim = animation.FuncAnimation(fig, update_plot, frames=amount,
            blit=False, repeat=True, interval=20)
    if save:
        writergif = animation.PillowWriter(fps=10) 
        anim.save('gif.gif', writer=writergif)
    else:
        plt.show()

    
def epoch_diff(noise=None, cutoff=0.5):
    if noise is None:
        noise = tf.random.normal([1, NOISE_DIM])
    def update_plot_epoch(epoch):
        checkpoint.restore(os.path.join(checkpoint_dir, f'ckpt-{epoch}'))
        generated_image = generator(noise, training=False)
        img = generated_image[0, :, :, :, 0].numpy()
        img = img > cutoff
        z, x, y = img.nonzero()
        plot._offsets3d = (x, y, z)
        ax.set_xlabel(f'epoch: {epoch}')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    data = np.zeros((RESOLUTION, RESOLUTION, RESOLUTION))
    z, x, y = data.nonzero()
    plot = ax.scatter(x, y, z, s=1)
    ax.set_xlim3d(0, RESOLUTION)
    ax.set_ylim3d(0, RESOLUTION)
    ax.set_zlim3d(0, RESOLUTION)
    anim = animation.FuncAnimation(fig, update_plot_epoch, frames=range(1,191,5),
            blit=False, repeat=True)
    writergif = animation.PillowWriter(fps=2) 
    anim.save('diff.gif', writer=writergif)


def disc_loss_graph(train_data, val_data):
    loss = []
    loss_val = []
    epochs = range(1, 63, 2)
    for epoch in epochs:
        print('epoch', epoch)
        checkpoint.restore(os.path.join(checkpoint_dir, f'ckpt-{epoch}'))
        with tf.GradientTape() as disc_tape:
            # training
            noise = tf.random.normal([int(BATCH_SIZE), NOISE_DIM])
            generated_images = generator(noise, training=False)
            real_output = discriminator(train_data, training=False)
            fake_output = discriminator(generated_images, training=False)
            disc_ws_loss = ws_disc_loss(real_output, fake_output)
            g_pen = gradient_penalty(
                train_data[0], generated_images, discriminator)
            disc_loss = disc_ws_loss + LAMBDA*g_pen
            loss.append(-1 * float(disc_loss))

            # val
            noise = tf.random.normal([int(BATCH_SIZE), NOISE_DIM])
            generated_images = generator(noise, training=False)
            real_output = discriminator(val_data, training=False)
            fake_output = discriminator(generated_images, training=False)
            disc_ws_loss = ws_disc_loss(real_output, fake_output)
            g_pen = gradient_penalty(
                val_data[0], generated_images, discriminator)
            disc_loss = disc_ws_loss + LAMBDA*g_pen
            loss_val.append(-1 * float(disc_loss))

    fig, ax = plt.subplots(1, 1, figsize=(10,5))
    ax.plot(epochs, loss, label='Training')
    ax.plot(epochs, loss_val, label='Validation')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    plt.show()


#plot_images(columns=1, rows=1, cutoff=0.5)
#animated_plot(amount=50, save=True, cutoff=0.5)
#snd_np = np.load('2.npy')
#snd = tf.convert_to_tensor(snd_np, dtype=tf.float32)
#epoch_diff(snd)
for b in train_ds:
    for b_val in val_ds:
        disc_loss_graph(b, b_val)
        break
    break
