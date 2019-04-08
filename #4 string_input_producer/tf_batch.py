# -*- coding: utf-8 -*-
import tensorflow as tf


"""
    tf.train.batch用来生成批次数据
"""

# 获取文件列表
# 对tf.train.match_filenames_once()所用到的变量是临时变量，和通过tf.Variable所初始化的变量有所不同
# 使用local_variables_initializer()初始化
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

example, label = features['i'], features['j']

# capacity，如果队列长度大于capacity时候，TensorFlow停止入队，否则入队
batch_size = 3
capacity = 1000 + 3 * batch_size

example_batch, label_batch = tf.train.batch(
    [example, label],
    batch_size = batch_size,
    capacity = capacity
)

with tf.Session() as sess:
    # 返回一个初始化所有局部变量的操作（Op）。初始化局部变量（GraphKeys.LOCAL_VARIABLE）。GraphKeys.LOCAL_VARIABLE中的变量指的是被添加入图中，但是未被储存的变量。local变量主要用作本地临时变量
    tf.local_variables_initializer().run() 
    # 添加节点用于初始化所有的变量(GraphKeys.VARIABLES)返回一个初始化所有全局变量的操作（Op）。在你构建完整个模型并在会话中加载模型后，运行这个节点。
    # tf.global_variables_initializer().run() 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(sess.run(files))

    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run([example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)
