# coding: utf-8
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil
import os
from tensorflow.examples.tutorials.mnist import input_data


def create_cnn_model(input_placeholder,training):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(input_placeholder, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=10,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    conv1b= tf.layers.batch_normalization(conv1,training=training)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1b, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
        filters=20,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    conv2b = tf.layers.batch_normalization(conv2,training=training)
    pool2 = tf.layers.max_pooling2d(inputs=conv2b,pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 20])
    dense = tf.layers.dense(inputs=pool2_flat,units=256, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=training) 
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)
    return logits
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/np.float32(255)
x_test = x_test/np.float32(255)

x = tf.placeholder(tf.float32, [None, 28,28],"input")
y_ = tf.placeholder(tf.int32, [None])
is_training = tf.placeholder(tf.bool)

with tf.variable_scope("model"):
    y1 = create_cnn_model(x,training=True)
with tf.variable_scope("model", reuse=True):
    y2 = create_cnn_model(x,training=False)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y1)

extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.cast(tf.argmax(y2, 1), tf.int32), y_)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

epoch_num = 10
batchsize = 100
N=x_train.shape[0]
for epoch in range(epoch_num) :
    T = time.time() 
    perm=np.random.permutation(N)
    for i in range(0,N,batchsize):
        
        batch_xs=x_train[perm[i:i+batchsize]]
        batch_ts=y_train[perm[i:i+batchsize]]
        
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ts})
        
    epochT = time.time()-T
    if os.path.exists("./models"):
        shutil.rmtree("./models")
    tf.saved_model.simple_save(sess, './models', inputs={"input": x}, outputs={"output": y2})
    
    loss, train_accuracy = sess.run([cross_entropy, accuracy], feed_dict={x: x_test, y_: y_test})
    print('Epoch: %d, Time :%.4f (s), Loss: %f,  Accuracy: %f' % (epoch + 1, epochT, loss, train_accuracy))








