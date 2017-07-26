# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

def visual_demo():
    mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
    batch_xs, batch_ys = mnist.train.next_batch(10)
    a = batch_xs[5]
    pixels = a.reshape((28, 28))
    plt.title('Label is {label}'.format(label=np.argmax(batch_ys[5])))
    plt.imshow(pixels, cmap='gray')
    plt.show()


def main():
    visual_demo()


if __name__=="__main__":
    main()