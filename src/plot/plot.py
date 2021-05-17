from scipy.interpolate import interp1d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os 
import time

from scipy.ndimage import zoom

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


#ds_files = tf.data.Dataset.list_files(
#    f'../data/{RESOLUTION}/*.npy', shuffle=True)
#dataset = ds_files.map(process_path)
#train_ds = dataset.shuffle(BUFFER_SIZE).batch(
#    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split
#
#
#ds_files_val = tf.data.Dataset.list_files(
#    f'../data/{RESOLUTION}_val/*.npy', shuffle=True)
#dataset_val = ds_files_val.map(process_path)
#val_ds = dataset.shuffle(BUFFER_SIZE).batch(
#    BATCH_SIZE, drop_remainder=True)  # drop if the data is not evenly split
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


def test_ploty(v, cutoff): 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(0, 4)
    ax.set_ylim3d(0, 4)
    ax.set_zlim3d(0, 4)

    axis = 2
    plane = 1
    plot_plane = True
    if plot_plane:
        for i in range(0,4):
            if i == plane:
                continue
            elif axis == 0:
                v[i, :, :] = 0
            elif axis == 1:
                v[:, i, :] = 0
            else:
                v[:, :, i] = 0

    cmap = plt.get_cmap('binary')(v)

    ax.voxels(v, facecolors=cmap)
    
    plt.show()


x = None
layer = discriminator.layers[0]
print(layer.name)
print(layer.get_weights()[0].shape)
# (filter_width, filter_height, filter_depth, input_channels, output_channels)
n_filters = layer.get_weights()[0].shape[-1]
n_filters = min(n_filters, 4)
#for i in range(n_filters):
#    weight = layer.get_weights()[0][:, :, :, :, i]
#    for j in range(weight.shape[-1]):
#        w = weight[:, :, :, j]
#        print(w.max())
#        test_ploty(w, 0)
#        time.sleep(0.1)


# plot first few filters
filters = layer.get_weights()[0]
# n_filters = outgoing channels
outgoing_channels = n_filters
ix = 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, :, i]
    # plot each channel separately
    # Range of incoming channels
    incoming_channels = f.shape[-1]
    incoming_channels= min(incoming_channels, 5)
    for j in range(incoming_channels):
        # Range of Depth of the kernel .i.e. 3
        Depth = 4
        for k in range(Depth):
            # specify subplot and turn of axis
            ax = plt.subplot((outgoing_channels*4), incoming_channels, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f[k, :, :,j], cmap='gray')
            ix += 1
# show the figure
plt.show()

#loss_plot(train_ds, val_ds, range(1,10))
#animated_plot(generator=generator, amount=50, save=False, cutoff=0.5)
#plot_images(generator=generator, cutoff=0.5)
#test_plot(generator, 0.5)
