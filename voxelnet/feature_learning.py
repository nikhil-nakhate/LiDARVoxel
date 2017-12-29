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

#Dimensions in metres
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

#For converting co ordinates to integer
#representing in a more scaled space
D_size = int(D * scale)
H_size = int(H * scale)
W_size = int(W * scale)

#Block Dim
D_ = int(D / vD)
H_ = int(H / vH)
W_ = int(W / vW)

#Grid Dim
Dg = int(D_size / D_)
Hg = int(H_size / H_)
Wg = int(W_size / W_)

print "Grid Dimensions: " +str(Dg) + ", " + str(Hg) + ", " + str(Wg)
print "Block Dimensions: " +str(D_) + ", " + str(H_) + ", " + str(W_)

T = 25

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

        #Here to Line 2018 is new. All old code is still here, but is commented out
        for x_ in x_batch:
            print "x_ size: " + str(x_.shape)
            print "x_: " + str(x_)

            pt_num = 0
            voxel_ID_list = [np.array([])]*300
            #voxel_ID_array = np.array([[x,y,z],[],[]],   [],   []) #ID*pt*(x,y,z)
            #sort the points into voxels
            for pt in x_:
                print "pt "+str(pt_num)+ ": " + str(pt)
                pt_num += 1
                #compute the grid and block indices
                gridX = pt[0] % Dg
                blockX = int(pt[0] / Dg)
                gridY = pt[1] % Hg
                blockY = int(pt[1] / Hg)
                gridZ = pt[2] % Wg
                blockZ = int(pt[2] / Wg)

                gridTuple = np.array([gridX, gridY, gridZ])

                voxelID = blockZ + blockY*W_ + blockX*W_*H_
                #print "blockX: "+str(blockX)+", blockY: "+str(blockY)+", blockZ: "+str(blockZ)
                #print "Voxel ID: " + str(voxelID)

                #add the point to the voxel list
                print "gridTuple: " + str(gridTuple)
                print "voxelList at id: " + str(voxel_ID_list[voxelID])
                print "Length: " + str(len(voxel_ID_list[voxelID]))
                if len(voxel_ID_list[voxelID]) == 0:
                    print "Length 0"
                    voxel_ID_list[voxelID] = [gridTuple]
                elif len(voxel_ID_list[voxelID]) == 1:
                    print "Length 1"
                    voxel_ID_list[voxelID] = np.concatenate((voxel_ID_list[voxelID], [gridTuple]), axis=0)
                else:
                    print "Length > 1"
                    voxel_ID_list[voxelID] = np.concatenate((voxel_ID_list[voxelID], [gridTuple]), axis=0)
                #voxel_ID_list[voxelID].append(gridTuple)
            for i in range(300):
                print "array of IDs [" + str(i) + "]: " + str(len(voxel_ID_list[i]))


            #description of what densex,y,z are
            # densex = tf.sparse_to_dense(sparse_indices=x_, output_shape=[D_size, H_size, W_size], sparse_values=x_[:, 0], validate_indices=False)
            # densey = tf.sparse_to_dense(sparse_indices=x_, output_shape=[D_size, H_size, W_size], sparse_values=x_[:, 1], validate_indices=False)
            # densez = tf.sparse_to_dense(sparse_indices=x_, output_shape=[D_size, H_size, W_size], sparse_values=x_[:, 2], validate_indices=False)

            # print "densex: " + str(densex)
            # print(session.run(tf.shape(densex)))
            #description of next few varibles
            # dense_batchx = tf.expand_dims(densex, 0)
            # dense_batchy = tf.expand_dims(densey, 0)
            # dense_batchz = tf.expand_dims(densez, 0)

            #description of next few varibles
            # sliced_x = tf.space_to_batch_nd(dense_batchx, block_shape=[D_, H_, W_], paddings=tf.zeros(shape=[3, 2], dtype=tf.int32))
            # sliced_y = tf.space_to_batch_nd(dense_batchy, block_shape=[D_, H_, W_], paddings=tf.zeros(shape=[3, 2], dtype=tf.int32))
            # sliced_z = tf.space_to_batch_nd(dense_batchz, block_shape=[D_, H_, W_], paddings=tf.zeros(shape=[3, 2], dtype=tf.int32))

            #description of next few varibles
            # blocked_x = tf.transpose(sliced_x, perm=[1, 2, 3, 0])
            # blocked_y = tf.transpose(sliced_y, perm=[1, 2, 3, 0])
            # blocked_z = tf.transpose(sliced_z, perm=[1, 2, 3, 0])

            # reduced mean for each array
            # subtract either scalar or repmat only from non zero, either threshold or find idxs
            # then concat final - - - 300 7

            #Not sure how to sample 25 so taking all 300
            # blocked_pt_x = tf.reshape(blocked_x, [Dg, Hg, Wg, D_, H_, W_])

            #zero = tf.constant(0, dtype=tf.int32)
            #where = tf.not_equal(blocked_pt_x, zero)

            #indices = np.where(where.eval(session=session))
            #indT = tf.constant(indices, shape=[6, 2500])
            #print(D_, H_, W_)
            #pos_featx = tf.reshape(indices, [Dg, Hg, Wg, D_, H_, W_])
            #print(session.run((indT)))
            # sparse = tf.SparseTensor(indices, tf.gather_nd(a_t, idx), a_t.get_shape())
            # dense = tf.sparse_tensor_to_dense(sparse)

            #just reshape to - - - 3 100
            #tf reduce mean
            #subtract scalar from each column
            #tf concat

            #print(session.run((pos_featx)))

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
