# -*- coding: utf-8 -*-
import tensorflow as tf

"""
    完整的输入数据处理框架
"""

# ===================================== #
# 读取TFRecord数据，还原图像数据
# ===================================== #
files = tf.train.match_filenames_once("./tfrecord/data.tfrecords-*")
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

features = tf.parse_single_example(
    serialized_example,
    features = {
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64),
    }
)

image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

# 从原始图像数据解析出像素矩阵，并根据图像尺寸还原图像
decoded_image = tf.decode_raw(image, tf.unit8)
decoded_image.set_shape([height, width, channels])

# ===================================== #
# 图像预处理
# ===================================== #
image_size = 299
distorted_image = preprocess_for_train(
    decoded_image,
    image_size,
    image_size, 
    None
)

min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label],
    batch_size = batch_size,
    capacity=capacity,
    min_after_dequeue = min_after_dequeue
)

# ===================================== #
# 定义网络、损失、优化等训练过程
# ===================================== #
learning_rate = 0.01
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run((tf.local_variables_initializer(),tf.global_variables_initializer()))
    # ===================================== #
    # 启动多个线程
    # ===================================== #
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    TRAINING_ROUNDS = 5000
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)

    coord.request_stop()
    coord.join(threads)