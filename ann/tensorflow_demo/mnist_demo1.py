# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import tempfile
import tf_cnnvis

def visual_demo(mnist):
    import matplotlib.pyplot as plt
    batch_xs, batch_ys = mnist.train.next_batch(10)
    a = batch_xs[5]
    pixels = a.reshape((28, 28))
    plt.title('Label is {label}'.format(label=np.argmax(batch_ys[5])))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x,[-1,28,28,1])

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)


    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([7*7*64,768])
        b_fc1 = bias_variable([768])

        h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([768,10])
        b_fc2 = bias_variable([10])
        y_conv = tf.matmul(h_fc1_drop,w_fc2) + b_fc2
    return x_image,y_conv,keep_prob


def main():
    mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
    # visual_demo(mnist)

    x = tf.placeholder(tf.float32,[None,784])

    y_ = tf.placeholder(tf.float32,[None,10])

    x_image,y_conv,keep_prob = deepnn(x)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
        correct_prediction = tf.cast(correct_prediction,tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print 'saving graph to:%s' % graph_location
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(700):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:batch[0],y_:batch[1],keep_prob:1.0
                })
                print 'step:%d,training accuracy:%g' % (i,train_accuracy)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        print batch
        print batch[0]
        feed_dict = {x: batch[0][1:2], y_: batch[1][1:2], keep_prob: 1.0}

    # deconv visualization
    layers = ["r", "p", "c"]
    total_time = 0

    import time
    start = time.time()
    # api call
    import tf_cnnvis
    is_success = tf_cnnvis.deconv_visualization(graph_or_path = tf.get_default_graph(), value_feed_dict = feed_dict,
                                      input_tensor=x_image, layers=layers, path_logdir="./Log/MNISTExample",
                                      path_outdir="./Output/MNISTExample")
    start = time.time() - start
    print("Total Time = %f" % (start))


def test():
    import time
    import sys
    for i in range(10):
        time.sleep(0.5)
        sys.stdout.write('\r%s' % i)
        sys.stdout.flush()


if __name__=="__main__":
    main()