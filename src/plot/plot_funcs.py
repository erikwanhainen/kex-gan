from scipy.interpolate import interp1d
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from src.model.utils import load_params
from src.model.losses import gradient_penalty, ws_disc_loss, ws_gen_loss

# CONSTANTS
params = load_params()

RESOLUTION = params['RESOLUTION']
BATCH_SIZE = params['BATCH_SIZE'] 
NOISE_DIM = params['NOISE_DIM']
LAMBDA = params['LAMBDA']


def plot(image, cut_off):
    image = image > cut_off
    figure = plt.figure()
    ax = figure.add_subplot(111, projection='3d')
    z, x, y = image.nonzero()
    ax.scatter(x, y, z)
    ax.set_xlim3d(0, RESOLUTION)
    ax.set_ylim3d(0, RESOLUTION)
    ax.set_zlim3d(0, RESOLUTION)
    plt.show()


def plot_images(generator, columns=5, rows=5, cutoff=0.4):
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
        fig = plt.figure()
        print('plotting')
        for i in range(columns*rows):
            noise = tf.random.normal([1, NOISE_DIM])
            generated_image = generator(noise, training=False)
            # IMG IS float32 type !
            img = generated_image[0, :, :, :, 0].numpy()
            img = img > cutoff
            z, x, y = img.nonzero()
            ax = fig.add_subplot(rows, columns, i+1, projection='3d')
            ax.set_xlim3d(0, RESOLUTION)
            ax.set_ylim3d(0, RESOLUTION)
            ax.set_zlim3d(0, RESOLUTION)
            ax.scatter(x, y, z, s=1)
        plt.show()


def animated_plot(generator, amount=1, save=False, cutoff=0.4):
    def update_plot(num):
        generated_image = generator(noise[num], training=False)
        img = generated_image[0, :, :, :, 0].numpy()
        img = img > cutoff
        z, x, y = img.nonzero()
        plot._offsets3d = (x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    noise = [tf.random.normal([1, NOISE_DIM]) for _ in range(amount)]
#    fst = np.random.random(NOISE_DIM)
#    snd = np.random.random(NOISE_DIM)
   # if two saved noise
#    fst_np = np.load('1.npy')
#    fst = tf.convert_to_tensor(fst_np, dtype=tf.float32)
#    snd_np = np.load('2.npy')
#    snd = tf.convert_to_tensor(snd_np, dtype=tf.float32)
#    linfit = interp1d([1, amount], np.vstack([fst, snd]), axis=0)
#    n = [i for i in range(1, amount+1)]
#    noise = []
#    arr = linfit(n)
#    for i in arr:
#        noise.append(np.reshape(i, (1, NOISE_DIM)))
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


def epoch_diff(generator, noise=None, cutoff=0.5):
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
    anim = animation.FuncAnimation(fig, update_plot_epoch, frames=range(1, 191, 5),
                                   blit=False, repeat=True)
    writergif = animation.PillowWriter(fps=2)
    anim.save('diff.gif', writer=writergif)


def disc_loss_graph(generator, discriminator, train_data, val_data):
    loss = []
    loss_val = []
    epochs = range(1, 61, 2)
    noise = tf.random.normal([int(BATCH_SIZE), NOISE_DIM])
    for epoch in epochs:
        print('epoch', epoch)
        checkpoint.restore(os.path.join(checkpoint_dir, f'ckpt-{epoch}'))
        with tf.GradientTape() as disc_tape:
            # training
            generated_images = generator(noise, training=False)
            real_output = discriminator(train_data, training=False)
            fake_output = discriminator(generated_images, training=False)
            disc_ws_loss = ws_disc_loss(real_output, fake_output)
            g_pen = gradient_penalty(
                train_data[0], generated_images, discriminator)
            disc_loss = disc_ws_loss + LAMBDA*g_pen
            loss.append(-1 * float(disc_loss))

            # val
            generated_images = generator(noise, training=False)
            real_output = discriminator(val_data, training=False)
            fake_output = discriminator(generated_images, training=False)
            disc_ws_loss = ws_disc_loss(real_output, fake_output)
            g_pen = gradient_penalty(
                val_data[0], generated_images, discriminator)
            disc_loss = disc_ws_loss + LAMBDA*g_pen
            loss_val.append(-1 * float(disc_loss))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(epochs, loss, label='Training')
    ax.plot(epochs, loss_val, label='Validation')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.legend()
    plt.show()


