import tensorflow as tf

# wrapper function
# These are Tensorflow's wrapper function for the most important routines we will be using in this implementation. 
# According to the paper, the variables are initialized with values from a normal distribution with mean of 0 and 
# standard deviation of 0.02. Both convolutions and transpose convolutions have 'same' padding and they both use 
# strides of 2 either to reduce in half or to double increase the inputs’ spatial dimensions.
def dense(x, out_units):
    return tf.layers.dense(x, out_units, activation=None,
                          kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

def conv2d(x, output_space):
    return tf.layers.conv2d(x, output_space, kernel_size=5, strides=2, padding="same", activation=None,
                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

def lrelu(x, alpha=0.2):
     # non-linear activation function
    return tf.maximum(alpha * x, x)

def batch_norm(x, training, epsilon=1e-5, momentum=0.9):
     return tf.layers.batch_normalization(x, training=training, epsilon=epsilon, momentum=momentum)
    
def transpose_conv2d(x, output_space):
    return tf.layers.conv2d_transpose(x, output_space, 5, strides=2, padding='same',
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

# Generator
# The network has 4 convolutional layers, all of them followed by batch normalization (except for the output layer) 
# and rectified linear unit (RELU) activations. It will take as input a random vector z (drawn from a normal distribution), 
# which will be reshaped in a 4D tensor and start a series of upsampling layers by applying transpose convolutional 
# operations with strides of 2.

# All the transpose convolutions use kernel filters of size 5x5 and the kernel depth goes from 512 all the way 
# down to 3 - representing the RGB color channels. The final layer then outputs a 32x32x3 tensor that will be 
# squashed between -1 and 1 through the Hyperbolic Tangent function.
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
    with tf.variable_scope('generator', reuse=reuse):
        fc1 = dense(z, 4*4*512)
        
        # Reshape it to start the convolutional stack
        fc1 = tf.reshape(fc1, (-1, 4, 4, 512))
        fc1 = batch_norm(fc1, training=training)
        fc1 = tf.nn.relu(fc1)
        
        t_conv1 = transpose_conv2d(fc1, 256)
        t_conv1 = batch_norm(t_conv1, training=training)
        t_conv1 = tf.nn.relu(t_conv1)
        
        t_conv2 = transpose_conv2d(t_conv1, 128)
        t_conv2 = batch_norm(t_conv2, training=training)
        t_conv2 = tf.nn.relu(t_conv2)
        
        logits = transpose_conv2d(t_conv2, output_dim)
        
        out = tf.tanh(logits)
        return out

# Discriminator
# The discriminator is also a 4 layer convolutional neural network followed by batch normalization 
# (except its input layer) and leaky RELU activations. The network receives a 32x32x3 image tensor 
# and performs regular convolutional operations with ‘same’ padding and strides of 2 - which basically 
# double the size of the filters at each layer. Finally, the discriminator needs to output probabilities. 
# For that, we use the Logistic Sigmoid activation function for the top layer.
def discriminator(x, reuse=False, alpha=0.2, training=True):
    """
    Discriminator network
    :param x: input for network
    :param reuse: Indicates whether or not the existing model variables should be used or recreated
    :param alpha: scalar for lrelu activation function
    :param training: Boolean for controlling the batch normalization statistics
    :return: A tuple of (sigmoid probabilities, logits)
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 32x32x?
        conv1 = conv2d(x, 64)
        conv1 = lrelu(conv1, alpha)
        
        conv2 = conv2d(conv1, 128)
        conv2 = batch_norm(conv2, training=training)
        conv2 = lrelu(conv2, alpha)
        
        conv3 = conv2d(conv2, 256)
        conv3 = batch_norm(conv3, training=training)
        conv3 = lrelu(conv3, alpha)

        # Flatten it
        flat = tf.reshape(conv3, (-1, 4*4*256))
        logits = dense(flat, 1)

        out = tf.sigmoid(logits)
        return out, logits