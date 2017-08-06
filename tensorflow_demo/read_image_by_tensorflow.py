#coding: utf-8
import tensorflow as tf
import sys
import os
from PIL import Image

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


  def return_data(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image



def main(_):
    filenames = ['/data/chenjianhong/cv/video_data/video/12719.jpg']
    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session() as sess:
            for i in range(1):
                # Read the filename:
                image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
                height, width = image_reader.read_image_dims(sess, image_data)
                print(height,width)
                print(image_reader.return_data(sess, image_data))
                a = image_reader.return_data(sess, image_data).tolist()
                x = list()
                for i in a:
                    for j in i:
                        x += j
                print(x[59000:59030])
                img = Image.fromarray(image_reader.return_data(sess, image_data), 'RGB')
                img.save("1-train.png")

                from scipy import misc
                b = misc.imread('/data/chenjianhong/cv/video_data/video/12719.jpg')
                a = b.tolist()
                x = list()
                for i in a:
                    for j in i:
                        x += j
                print(x[59000:59030])
                img = Image.fromarray(b, 'RGB')
                img.save("2-train.png")

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])