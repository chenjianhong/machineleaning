#coding: utf-8
import os
import sys
import tensorflow as tf

def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'image/class/label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image/encoded'], tf.uint8)
  # image = tf.reshape(image, tf.stack([299, 299, 1]))


  # from tensorflow.examples.tutorials.mnist import mnist
  # image.set_shape([mnist.IMAGE_PIXELS])

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['image/class/label'], tf.int32)

  return image, label


def inputs(train):
  """Reads input data num_epochs times.
  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.
  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  num_epochs = None
  filename = r'E:\SecureCRT\download\videos_validation_00004-of-00005.tfrecord'
  filename = r'E:\SecureCRT\download\videos_validation_00003-of-00005.tfrecord'

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs=num_epochs)

    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)

    print(image)

    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    # images, sparse_labels = tf.train.shuffle_batch(
    #     [image, label], batch_size=10, num_threads=1,
    #     capacity=20,
    #     # Ensures a minimum amount of shuffling of examples.
    #     min_after_dequeue=10)

    return image, label

def main(_):
    with tf.Graph().as_default():
        images, labels = inputs(train=True)
        print(images,labels)
        sess = tf.Session()

        # Initialize the variables (the trained variables and the
        # epoch counter).
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(images)

        for i in range(5):
            example, l = sess.run([images, labels])
            # img = Image.fromarray(example, 'RGB')
            # img.save("output/" + str(i) + '-train.png')

            print(example, l)
            print(example.shape)

if __name__ == '__main__':
    tf.app.run(main=main,argv=[sys.argv[0]])