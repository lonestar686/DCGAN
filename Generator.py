# generator for dcgan
from network_modules import Module
from network_keras import dense, conv2d, transpose_conv2d, lrelu
from network_keras import batch_norm, reshape, activation

import tensorflow as tf

def generator(z, output_dim, reuse=False, alpha=0.2, training=True):
    """
    Generator network
    :param z: input random vector z
    :param output_dim: output dimension of the network
    :param reuse: Indicates whether or not the existing model variables should be used or recreated
    :param alpha: scalar for lrelu activation function
    :param training: Boolean for controlling the batch normalization statistics
    :return: model's output
    """
    g = Generator(output_dim)

    return g(z)

class Generator(Module):
    """ generator
        The network has 4 convolutional layers, all of them followed by batch normalization 
        (except for the output layer) and rectified linear unit (RELU) activations. It will 
        take as input a random vector z (drawn from a normal distribution), which will be reshaped 
        in a 4D tensor and start a series of upsampling layers by applying transpose convolutional 
        operations with strides of 2.

        All the transpose convolutions use kernel filters of size 5x5 and the kernel depth goes 
        from 512 all the way down to 3 - representing the RGB color channels. The final layer then 
        outputs a 32x32x3 tensor that will be squashed between -1 and 1 through the 
        Hyperbolic Tangent function.
    """
    def __init__(self, output_dim):

        with tf.variable_scope('generator'):

            self.dense = dense(4*4*512)
            
            # Reshape it to start the convolutional stack
            self.reshape = reshape((4, 4, 512))
            self.batch_norm1 = batch_norm()
            self.relu1 = activation('relu')
            
            self.transpose_conv2d_2 = transpose_conv2d(256)
            self.batch_norm2 = batch_norm()
            self.relu2 = activation('relu')
            
            self.transpose_conv2d_3 = transpose_conv2d(128)
            self.batch_norm3 = batch_norm()
            self.relu3 = activation('relu')
            
            self.transpose_conv2d_4= transpose_conv2d(output_dim)
            
            self.tanh = activation('tanh')

    def forward(self, z):

        fc1 = self.dense(z)
        
        # Reshape it to start the convolutional stack
        fc1 = self.reshape(fc1)
        fc1 = self.batch_norm1(fc1)
        fc1 = self.relu1(fc1)
        
        t_conv1 = self.transpose_conv2d_2(fc1)
        t_conv1 = self.batch_norm2(t_conv1)
        t_conv1 = self.relu2(t_conv1)
        
        t_conv2 = self.transpose_conv2d_3(t_conv1)
        t_conv2 = self.batch_norm3(t_conv2)
        t_conv2 = self.relu3(t_conv2)
        
        logits = self.transpose_conv2d_4(t_conv2)
        
        out = self.tanh(logits)

        return out

