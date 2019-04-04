# coding: utf-8
import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
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
# MNISTデータのダウンロード
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train/np.float32(255)
x_test = x_test/np.float32(255)

# 入力データを入れる箱
x = tf.placeholder(tf.float32, [None, 28,28])
# 正解ラベルを入れる箱
y_ = tf.placeholder(tf.int32, [None])
is_training = tf.placeholder(tf.bool)

# モデルの構築
#y = create_model(x, input_size, hidden_size, output_size)
y = create_cnn_model(x,is_training)

# 損失関数
cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)
# 学習用モデル

extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_ops):
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# 予測
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
# 認識精度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# セッション
sess = tf.InteractiveSession()
# 初期化
tf.global_variables_initializer().run()

# 学習用のパラメータ設定
epoch_num = 10
batchsize = 100
N=x_train.shape[0]
saver = tf.train.Saver()
# 学習
for epoch in range(epoch_num) :
    T = time.time() # 処理時間計測用
    perm=np.random.permutation(N)
    for i in range(0,N,batchsize):
        # 学習用データのセット
        batch_xs=x_train[perm[i:i+batchsize]]
        batch_ts=y_train[perm[i:i+batchsize]]
        # 学習実行

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ts,is_training:True})
        
    epochT = time.time()-T

    # 学習モデルの評価
    loss, train_accuracy,py = sess.run([cross_entropy, accuracy,y], feed_dict={x: x_test, y_: y_test,is_training:False})
    print('Epoch: %d, Time :%.4f (s), Loss: %f,  Accuracy: %f' % (epoch + 1, epochT, loss, train_accuracy))
saver.save(sess, "result/model.ckpt")


'''
### 学習させたモデルを使用して予測を行う ###
# テストデータからランダムに抜き出す
idx = np.random.choice(len(x_test), 1)
for i in idx:
    test_input = x_test[i][np.newaxis, :]
    # 予測結果を取得
    pre = np.argmax(sess.run(y, feed_dict={x: test_input,is_training:False}))
    # ラベルデータ
    test_y = y_test[i]
    # ラベルデータと予測結果が異なっている場合は入力データ(画像データ)を出力
    if(pre != test_y):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        # ファイル名：label_(ラベル値)_predict_(予測値).png
        file_name = "label_" + str(test_y) + "_predict_" + str(pre) + ".png"
        plt.savefig(file_name)

'''








