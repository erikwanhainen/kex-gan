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


ds_files = tf.data.Dataset.list_files(
    f'../data/{RESOLUTION}/*.npy', shuffle=True)
dataset = ds_files.map(process_path)
train_ds = dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split


ds_files_val = tf.data.Dataset.list_files(
    f'../data/{RESOLUTION}_val/*.npy', shuffle=True)
dataset_val = ds_files_val.map(process_path)
val_ds = dataset.shuffle(BUFFER_SIZE).batch(
    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split
# Create distributed dataset depending on strategy


if RESTORE:
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    print('Restoring last model')



def loss_plot(tds, vds, epochs):
    big_boy_t = [] 
    big_boy_v = []
    count = 0
    for e in epochs:
        t_loss = []
        v_loss = []
        checkpoint.restore(os.path.join(checkpoint_dir, f'ckpt-{e}'))
        for batch in tds:
            loss = disc_loss_batch(generator, discriminator, batch).numpy()
            print(type(loss))
            t_loss.append(loss)
            count += 1
            if count == 7:
                print('BREAKING')
                count = 0
                break


        for batch in vds:
            loss = disc_loss_batch(generator, discriminator, batch).numpy()
            v_loss.append(loss)

        big_boy_t.append(sum(t_loss)/len(t_loss))
        big_boy_v.append(sum(v_loss)/len(v_loss))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(epochs, big_boy_t, label='Training')
    ax.plot(epochs, big_boy_v, label='Validation')
    ax.set_ylabel('negative disc loss')
    ax.set_xlabel('epoch')
    ax.legend()
    plt.savefig('nice.png')

loss_plot(train_ds, val_ds, range(1,10))

#animated_plot(generator=generator, amount=50, save=False, cutoff=0.5)
# plot_images(generator=generator, cutoff=0.5)
# test_plot(generator, 0.5)
