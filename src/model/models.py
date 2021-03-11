import tensorflow as tf

def Deconv(inputs, f_dim_in, dim, net, batch_size, f_dim_out = None, stride = 2 ):
	if f_dim_out is None: 
		f_dim_out = f_dim_in/2 
	return tl.layers.DeConv3dLayer(inputs,
                shape = [4, 4, 4, f_dim_out, f_dim_in],
                output_shape = [batch_size, dim, dim, dim, f_dim_out],
                strides=[1, stride, stride, stride, 1],
                W_init = tf.random_normal_initializer(stddev=0.02),
                act=tf.identity, name='g/net_' + net + '/deconv')

def Conv3D(inputs, f_dim_out, net, f_dim_in = None, batch_norm = False, is_train = True):
	if f_dim_in is None: 
		f_dim_in = f_dim_out/2
	layer = tl.layers.Conv3dLayer(inputs, 
                shape=[4, 4, 4, f_dim_in, f_dim_out],
                W_init = tf.random_normal_initializer(stddev=0.02),
                strides=[1, 2, 2, 2, 1], name= 'd/net_' + net + '/conv')


def generator(inputs, batch_size):
    output_size, half, forth, eighth, sixteenth = 128, 64, 32, 16, 8
    gf_dim = 256 # Dimension of gen filters in first conv laye
