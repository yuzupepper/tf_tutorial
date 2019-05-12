# coding: utf-8
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/np.float32(255)
x_test = x_test/np.float32(255)


with tf.Session(graph=tf.Graph()) as sess:
    y_ = tf.placeholder(tf.int32, [None])

    meta_graph=tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "./models")
    model_signature = meta_graph.signature_def['serving_default']
    # input
    x = sess.graph.get_tensor_by_name(model_signature.inputs['input'].name)
    # output
    y = sess.graph.get_tensor_by_name(model_signature.outputs['output'].name)
    # predict
    correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
    # accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = sess.run([accuracy], feed_dict={x: x_test, y_: y_test})
    print(acc)













