# discriminator for dcgan
from network_modules import Module
from network_keras import dense, conv2d, lrelu
from network_keras import batch_norm, flatten, activation

import tensorflow as tf

def discriminator(x, reuse=False, alpha=0.2, training=True):
    """
    Discriminator network
    :param x: input for network
    :param reuse: Indicates whether or not the existing model variables should be used or recreated
    :param alpha: scalar for lrelu activation function
    :param training: Boolean for controlling the batch normalization statistics
    :return: A tuple of (sigmoid probabilities, logits)
    """
 #TODO: not working, will fix it tomorrow.
    d = Discriminator(alpha)
    return d(x)
#
class Discriminator(Module):
    """ Discriminator
        The discriminator is also a 4 layer convolutional neural network followed by batch normalization 
        (except its input layer) and leaky RELU activations. The network receives a 32x32x3 image tensor 
        and performs regular convolutional operations with ‘same’ padding and strides of 2 - which basically 
        double the size of the filters at each layer. Finally, the discriminator needs to output probabilities. 
        For that, we use the Logistic Sigmoid activation function for the top layer.
    """
    def __init__(self, alpha):

        with tf.variable_scope('discriminator'):
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

        print(tf.trainable_variables())

    def forward(self, x):

        # Input layer is 32x32x?
        conv1 = self.conv1(x)
        conv1 = self.lrelu1(conv1)
        
        conv2 = self.conv2(conv1)
        conv2 = self.batch_norm2(conv2)
        conv2 = self.lrelu2(conv2)
        
        conv3 = self.conv3(conv2)
        conv3 = self.batch_norm3(conv3)
        conv3 = self.lrelu3(conv3)

        # Flatten it
        #flat = tf.reshape(conv3, (-1, 4*4*256))
        flat   = self.flatten(conv3)
        logits = self.dense(flat)

        out = self.sigmoid(logits)

        return out, logits
