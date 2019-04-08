import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


mnist = input_data.read_data_sets(
    "./mnist_data/",
    one_hot = True
)
images = mnist.train.images
labels = mnist.train.labels                # 训练样本的label作为一个属性保存到TFRecord
pixels = images.shape[1]                   # 图像分辨率作为一个属性保存
num_examples = mnist.train.num_examples    # 图片数量


filename="./tfrecord/output.tfrecords"           # TFRecord保存位置
writer = tf.python_io.TFRecordWriter(filename)    # 创建一个writer
for index in range(num_examples):
    image_raw = images[index].tostring()          # 将图像矩阵转为字符串
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(labels[index])),
        'image_raw': _bytes_feature(image_raw)
    }))
    writer.write(example.SerializeToString())     # 将一个Example保存到TFRecord中
writer.close()
