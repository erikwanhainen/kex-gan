import tensorflow as tf


from src.model.utils import load_params

params = load_params()

BATCH_SIZE = params['BATCH_SIZE']
NUM_GPUS = params['NUM_GPUS']




def gradient_penalty(real_data, fake_data, disc):
    """
      real_data: shapes from batch
      fake_data: generated samples
    """
    bs = int(BATCH_SIZE/NUM_GPUS)
    alpha = tf.random.uniform(shape=[bs, 1, 1, 1, 1], minval=0., maxval=1.)
    difference = fake_data - real_data
    interpolates = real_data + (alpha*difference)

    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        prediction = disc(interpolates)
    gradients = tape.gradient(prediction, interpolates)


    slopes = tf.sqrt(tf.reduce_sum(
        gradients ** 2, axis=[1, 2, 3, 4]))  # Try this

    gradient_penalty = tf.reduce_mean((slopes-1.)**2)

    return gradient_penalty

def ws_disc_loss(real_output, fake_output):
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


def ws_gen_loss(fake_output):
    return -tf.reduce_mean(fake_output)

