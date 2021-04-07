#from google.colab import drive
#drive.mount('/content/drive')

import tensorflow as tf 
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import activations

import os
import time
#

#### CONSTANTS 
RESOLUTION = 128
BUFFER_SIZE = 30 # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
BATCH_SIZE = 15
NOISE_DIM = 200
RESTORE = False
NUM_DISC_UPDATES = 5
LAMBDA = 10

##### CREATE DATASET FROM NPY FILES ##########
ds_files = tf.data.Dataset.list_files(f'./src/data/{RESOLUTION}/*.npy', shuffle = True)

def read_npy_file(filename):
    data = np.load(filename.numpy().decode()).reshape(RESOLUTION,RESOLUTION,RESOLUTION,1)
    return data.astype(np.float32)

def process_path(file_path):
  """
    Read npy file 
  """
  image = tf.py_function(read_npy_file, [file_path],[tf.float32,]) 
  return image

dataset = ds_files.map(process_path)

# Fills up a buffer of images that the batch is then drawn from


train_ds = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True) # drop if the data is not evenly split

#########################################
# Define generator and discriminator models

def generator_model():
  """
    bruh
  """
  filters_initial = 256
  kernel_size = (4, 4, 4)
  half, fourth, eighth, sixteenth, threetwo = int(RESOLUTION / 2), int(RESOLUTION / 4), int(RESOLUTION / 8), int(RESOLUTION / 16), int(RESOLUTION / 32)

  model = tf.keras.Sequential()

  model.add(layers.Dense(threetwo*threetwo*threetwo*filters_initial, use_bias=False, input_shape=(NOISE_DIM, ))) # Latent vector z 200 dimensional
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Reshape((threetwo, threetwo, threetwo, filters_initial)))
  assert model.output_shape == (None, threetwo, threetwo, threetwo, filters_initial)

  model.add(layers.Conv3DTranspose(filters_initial, kernel_size, strides=(2, 2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, sixteenth, sixteenth, sixteenth, filters_initial)
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Conv3DTranspose(filters_initial / 2, kernel_size, strides=(2, 2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, eighth, eighth, eighth, filters_initial/2)
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Conv3DTranspose(filters_initial / 4, kernel_size, strides=(2, 2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, fourth, fourth, fourth, filters_initial/4)
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Conv3DTranspose(filters_initial / 8, kernel_size, strides=(2, 2, 2), padding='same', use_bias=False))
  assert model.output_shape == (None, half, half, half, filters_initial/8)
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())

  model.add(layers.Conv3DTranspose(1, kernel_size, strides=(2, 2, 2), padding='same', use_bias=False, activation='tanh'))
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
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(filters_initial / 4, kernel_size, strides=(2, 2, 2), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(filters_initial/2, kernel_size, strides=(2, 2, 2), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(filters_initial, kernel_size, strides=(2, 2, 2), padding='same'))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv3D(1, kernel_size, strides=(1, 1, 1), padding='same'))

    #model.add(layers.Activation(activations.sigmoid))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='linear'))


    return model

generator = generator_model()

noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)

discriminator = discriminator_model()
decision = discriminator(generated_image)
print(decision)

import matplotlib.pyplot as plt

#img = generated_image[0, :, :, :, 0].numpy() # IMG IS float32 type !

#plot(img)

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

###### THE LOSS FUNCTION (STANDARD FORMULATION) #####
def gradient_penalty(real_data, fake_data, disc):
  """
    real_data: shapes from batch
    fake_data: generated samples
  """
  alpha = tf.random.uniform(shape=[BATCH_SIZE, 1], minval=0., maxval=1.)
  difference = fake_data - real_data
  inter = []
  for i in range(BATCH_SIZE):
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

  slopes = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1])) # Try this
  gradient_penalty = tf.reduce_mean((slopes-1.)**2)
  return gradient_penalty

def ws_disc_loss(real_output, fake_output):
  return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

def ws_gen_loss(fake_output):
  return -tf.reduce_mean(fake_output)

## Define optimizer  ##
# parameters taken from https://github.com/igul222/improved_wgan_training/blob/master/gan_toy.py
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)  
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

##### Save and restore model during training #####

checkpoint_dir = './src/checkpoints/w_128'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

## TRAINING LOOP
EPOCHS = 60


@tf.function
def train_step(images, disc_updates):
  """
    NOTE: we can change how often the generator is updated vs discriminator
  """

  for k in range(disc_updates):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True) # Should training be false here?

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      #disc_loss = discriminator_loss(real_output, fake_output)
      disc_ws_loss = ws_disc_loss(real_output, fake_output)
      g_pen = gradient_penalty(images[0], generated_images, discriminator)
      disc_loss = disc_ws_loss + LAMBDA*g_pen

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

  noise = tf.random.normal([BATCH_SIZE, NOISE_DIM]) # Sample new noise
  with tf.GradientTape() as gen_tape:
    generated_images = generator(noise, training=True)

    fake_output = discriminator(generated_images, training=True) # training false?

    #gen_loss = generator_loss(fake_output)
    gen_loss = ws_gen_loss(fake_output)

  gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
  generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))


def train(dataset, epochs):
  gen_amount = 10
  noise = tf.random.normal([gen_amount, NOISE_DIM])
  for epoch in range(epochs):
    start = time.time()
    counter = 0
    for image_batch in dataset:
      train_step(image_batch, NUM_DISC_UPDATES)
      counter += BATCH_SIZE
      print(counter)

    # Print avg discriminator of image_batch

    if 1 == 1:
        checkpoint.save(file_prefix = checkpoint_prefix)

    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print('PRINTING DISCRIMINATOR THINGS :D ')
    generated_image = generator(noise, training=False)
    print(discriminator(generated_image))

    for images in dataset.take(1):
      numpy_images = images[0].numpy()
      print(discriminator(numpy_images, training=False))

    for i in range(gen_amount):
      img = generated_image[i, :, :, :, 0].numpy() # IMG IS float32 type !
      plot(img, 0.4) # cut off = 0.4
    # Print discriminator of generated_images :)

  ## Display image gif thingy

print("Starting train")
if RESTORE:
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  print('Restoring last model')
train(train_ds, EPOCHS)

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

gen_amount = 10
cut_off = 0.4
noise = tf.random.normal([gen_amount, NOISE_DIM])
generated_image = generator(noise, training=False)


#print(discriminator(generated_image, training=False))

print('random real')
for images in train_ds.take(1):  # only take first element of dataset
  numpy_images = images[0].numpy()
  #print(discriminator(numpy_images, training=False))
  plot(numpy_images[0,:,:,:,0], cut_off)

print('generated')
for i in range(gen_amount):
  img = generated_image[i, :, :, :, 0].numpy() #
  plot(img, cut_off)
