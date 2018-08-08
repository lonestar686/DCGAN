#
import tensorflow as tf

# wrapper function
# These are Tensorflow's wrapper function for the most important routines we will be using in this implementation. 
# According to the paper, the variables are initialized with values from a normal distribution with mean of 0 and 
# standard deviation of 0.02. Both convolutions and transpose convolutions have 'same' padding and they both use 
# strides of 2 either to reduce in half or to double increase the inputs’ spatial dimensions.
def dense(out_units, activation=None):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    return tf.keras.layers.Dense(out_units, activation=activation,
                                 kernel_initializer=initializer)

def conv2d(output_space, kernel_size=(5,5), strides=(2,2), activation=None): 
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    return tf.keras.layers.Conv2D(output_space, 
                                    kernel_size=kernel_size, strides=strides, 
                                    padding="same", activation=activation, 
                                    kernel_initializer=initializer)

def transpose_conv2d(output_space, kernel_size=(5,5), strides=(2,2), activation=None):
    initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.02)
    return tf.keras.layers.Conv2DTranspose(output_space, 
                                            kernel_size=kernel_size, strides=strides, 
                                            padding='same',
                                            kernel_initializer=initializer)

def lrelu(alpha=0.2):
     # non-linear activation function
    return tf.keras.layers.LeakyReLU(alpha=alpha)

def batch_norm(epsilon=1e-5, momentum=0.9):
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    return tf.keras.layers.BatchNormalization(epsilon=epsilon, momentum=momentum, 
                                                gamma_initializer=initializer)
    
def activation(act):
    return tf.keras.layers.Activation(act)

def identity():
    return tf.keras.layers.Lambda(lambda x: x)


def dropout(a):
    return tf.keras.layers.Dropout(a)

def concat(axis=-1):
    return tf.keras.layers.Concatenate(axis)

def flatten():
    return tf.keras.layers.Flatten()

def reshape(target_shape):
    return tf.keras.layers.Reshape(target_shape)
