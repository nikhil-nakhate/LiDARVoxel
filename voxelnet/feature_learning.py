from utils import dataset as dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np

# Adding Seed so that random initialization is consistent
from numpy.random import seed


#Path
train_path = "data/example_data/"

#Just return sequence for ref
#pointclouds, positions, orientations, labels
#Using NCHW format

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

batch_size = 1

scale = 100

D = 4.5
H = 4.0
W = 4.0

#origin shift
shiftX = D
shiftY = W / 2
shiftZ = H / 2

#Voxel Dim
vD = 1.5
vH = 0.4
vW = 0.4

#Grid Dim
D_ = int(D / vD)
H_ = int(H / vH)
W_ = int(W / vW)

T = 25

#For converting co ordinates to integer
#representing in a more scaled space
D_size = int(D * scale)
H_size = int(H * scale)
W_size = int(W * scale)

#Check for index out of bounds since we are multiplying by 100 and putting values in indces
#Assuming here that all the points are at equal intensity that is 1

# Prepare input data
classes = ['Box', 'Sphere']
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2

#Final feature size
m = 128


# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(train_path, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))


session = tf.Session()
#pc_tensor = tf.convert_to_tensor(data.train.pointclouds[0], dtype=tf.int32)
x = tf.placeholder(tf.float32, shape=[None, D_size, H_size, W_size], name='x')

## labels
# y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
# y_true_cls = tf.argmax(y_true, dimension=1)


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):

    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])

    biases = create_biases(num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()

    num_features = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

total_iterations = 0

#saver = tf.train.Saver()


def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_positions, y_orientations, train_labels = data.train.next_batch(batch_size)
        #x_valid_batch, y_valid_positions, y_valid_orientations, valid_labels = data.valid.next_batch(batch_size)

        for x_ in x_batch:
            densex = tf.sparse_to_dense(sparse_indices=x_, output_shape=[D_size, H_size, W_size], sparse_values=1,validate_indices=False)
            # print(session.run(tf.shape(densex)))
            dense_batchx = tf.expand_dims(densex, 0)
            sliced_x = tf.space_to_batch_nd(dense_batchx, block_shape=[D_, H_, W_], paddings=tf.zeros(shape=[3, 2], dtype=tf.int32))
            print(session.run(tf.shape(sliced_x)))

        #print(len(x_batch_new))
        # feed_dict_tr = {x: x_batch,
        #                 y_true: y_true_batch}
        # feed_dict_val = {x: x_valid_batch,
        #                  y_true: y_valid_batch}
        #
        # session.run(optimizer, feed_dict=feed_dict_tr)
        #
        # if i % int(data.train.num_examples / batch_size) == 0:
        #     val_loss = session.run(cost, feed_dict=feed_dict_val)
        #     epoch = int(i / int(data.train.num_examples / batch_size))
        #
        #     show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
        #     saver.save(session, 'dogs-cats-model')

    total_iterations += num_iteration

train(num_iteration=1)

#pointclouds, positions, orientations, labels