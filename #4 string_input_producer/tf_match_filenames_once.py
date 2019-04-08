# -*- coding: utf-8 -*-
import tensorflow as tf


"""
    tf.train.match_filenames_once获取符合正则表达式的所有文件
"""

# 获取文件列表
files = tf.train.match_filenames_once("./tfrecord/data.tfrecords-*")

# string_input_producer创建输入队列
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 解析样本
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    }
)

with tf.Session() as sess:
    # 使用 tf.train.match_filenames_once需要初始化变量
    tf.local_variables_initializer().run()
    print(sess.run(files))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(6):
        print(sess.run([features['i'], features['j']]))

    coord.request_stop()
    coord.join(threads)