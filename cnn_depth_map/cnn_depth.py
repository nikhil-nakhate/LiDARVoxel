from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

#import read data function
import read_data as rd

tf.logging.set_verbosity(tf.logging.INFO)


#Neural Net Parameters - can vary these to see the effects
conv1kernel = 5
conv1filters = 8
pool1dim = 2
conv2kernel = 5
conv2filters = 16
pool2dim = 2
learnRate = 0.001
batchSize = 10
numIters = 500
dropoutRate = 0.0

# Our application logic will be added here

num_voxels = 300
max_pts = 30


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""

  #feature learning layers
  input_layer = tf.reshape(features["x"], [-1,50,50,1])

  # Convolutional Layer #1
  #print("conv1 layer")
  conv3 = tf.layers.conv2d(
      inputs=input_layer,
      filters=conv1filters,
      strides=2,
      kernel_size=16,
      padding="same",
      activation=tf.nn.relu)

  print("shape after conv3: "+str(conv3.shape))

  # Pooling Layer #1
  #print("pool1 layer")
  pool1 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[pool1dim, pool1dim], strides=2)
  print("shape after pool1: "+str(pool1.shape))
  # Convolutional Layer #2 and Pooling Layer #2
  #print("conv2 layer")
  conv4 = tf.layers.conv2d(
      inputs=pool1,
      filters=conv2filters,
      strides=2,
      kernel_size=32,
      padding="same",
      activation=tf.nn.relu)
  print("shape after conv4: "+str(conv4.shape))

  #print("pool2 layer")
  pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[pool2dim, pool2dim], strides=2)
  print("shape after pool2: "+str(pool2.shape))
  #print('pool2 shape: ' + str(pool2.shape))

  # Dense Layer
  #print("pool2 flatten")
  pool2shape = pool2.shape
  pool2_flat = tf.reshape(pool2, [-1, pool2shape[1]*pool2shape[2]*pool2shape[3]])
  #print("dense layer")
  dense = tf.layers.dense(inputs=pool2_flat, units=256, activation=tf.nn.relu)
  print("shape after dense: "+str(dense.shape))
  #print("perform dropouts")
  dropout = tf.layers.dropout(
      inputs=dense, rate=dropoutRate, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits Layer
  #print("logits layer")
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=2)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learnRate)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":
    # Load training and eval data
    # mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    train_data, train_labels, eval_data, eval_labels = rd.get_depth_data_sets()

    # train_data = mnist.train.images # Returns np.array
    # train_data = np.random.rand(100,2500).astype(np.float32)
    print("training: " + str(train_data))
    # #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # train_labels = np.asarray(np.random.randint(2, size=100), dtype=np.int32)
    print("train_labels: " + str(train_labels))
    # #eval_data = mnist.test.images # Returns np.array
    # eval_data = np.random.rand(10,2500).astype(np.float32)
    print("eval_data: " + str(eval_data))
    # #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # eval_labels = np.asarray(np.random.randint(2, size=10), dtype=np.int32)
    print("eval_labels: " + str(eval_labels))

    # Create the Estimator
    lidar_classifier = tf.estimator.Estimator(
        # model_fn=cnn_model_fn, model_dir="/tmp/lidar_convnet_model")
        model_fn=cnn_model_fn)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batchSize,
        num_epochs=None,
        shuffle=True)
    lidar_classifier.train(
        input_fn=train_input_fn,
        steps=numIters,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = lidar_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
