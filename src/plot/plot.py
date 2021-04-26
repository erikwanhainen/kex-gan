from scipy.interpolate import interp1d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os 

from tensorflow.keras import layers
from tensorflow.keras import activations


from src.model.models import generator_model, discriminator_model
from src.plot.plot_funcs import animated_plot, plot_images, test_plot


# CONSTANTS
RESTORE = True


##### Save and restore model during training #####
checkpoint_dir = './src/checkpoints/w_128'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


generator = generator_model()
discriminator = discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(
    1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(
    1e-4, beta_1=0.5, beta_2=0.9)



checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

if RESTORE:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    print('Restoring last model')

#animated_plot(generator=generator, amount=50, save=False, cutoff=0.5)
# plot_images(generator=generator, cutoff=0.5)
test_plot(generator, 0.5)
