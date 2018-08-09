# discriminator for dcgan
from network_modules import Module
from network_keras import dense, conv2d, lrelu
from network_keras import batch_norm, flatten, activation

import tensorflow as tf

class Discriminator(Module):
    """ Discriminator
        The discriminator is also a 4 layer convolutional neural network followed by batch normalization 
        (except its input layer) and leaky RELU activations. The network receives a 32x32x3 image tensor 
        and performs regular convolutional operations with ‘same’ padding and strides of 2 - which basically 
        double the size of the filters at each layer. Finally, the discriminator needs to output probabilities. 
        For that, we use the Logistic Sigmoid activation function for the top layer.
    """
    def __init__(self, alpha):

        self.conv1 = conv2d(64)
        self.lrelu1 = lrelu(alpha)
        
        self.conv2 = conv2d(128)
        self.batch_norm2 = batch_norm()
        self.lrelu2 = lrelu(alpha)
        
        self.conv3 = conv2d(256)
        self.batch_norm3 = batch_norm()
        self.lrelu3 = lrelu(alpha)

        self.flatten = flatten()
        self.dense = dense(1)
        self.sigmoid = activation('sigmoid')

    def forward(self, x, reuse, training):

        with tf.variable_scope('discriminator'):
            # Input layer is 32x32x?
            conv1 = self.conv1(x)
            conv1 = self.lrelu1(conv1)
            
            conv2 = self.conv2(conv1)
            conv2 = self.batch_norm2(conv2, training=training)
            conv2 = self.lrelu2(conv2)
            
            conv3 = self.conv3(conv2)
            conv3 = self.batch_norm3(conv3, training=training)
            conv3 = self.lrelu3(conv3)

            # Flatten it
            #flat = tf.reshape(conv3, (-1, 4*4*256))
            flat   = self.flatten(conv3)
            logits = self.dense(flat)

            out = self.sigmoid(logits)

        return out, logits
