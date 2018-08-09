# generator for dcgan
from network_modules import Module
from network_keras import dense, conv2d, transpose_conv2d, lrelu
from network_keras import batch_norm, reshape, activation

import tensorflow as tf

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

    def forward(self, z, reuse, training):

        with tf.variable_scope('generator', reuse=reuse):

            fc1 = self.dense(z)
            
            # Reshape it to start the convolutional stack
            fc1 = self.reshape(fc1)
            fc1 = self.batch_norm1(fc1, training=training)
            fc1 = self.relu1(fc1)
            
            t_conv1 = self.transpose_conv2d_2(fc1)
            t_conv1 = self.batch_norm2(t_conv1, training=training)
            t_conv1 = self.relu2(t_conv1)
            
            t_conv2 = self.transpose_conv2d_3(t_conv1)
            t_conv2 = self.batch_norm3(t_conv2, training=training)
            t_conv2 = self.relu3(t_conv2)
            
            logits = self.transpose_conv2d_4(t_conv2)
            
            out = self.tanh(logits)

        return out

