# -*- coding: utf-8 -*-
import tensorflow as tf


queue = tf.FIFOQueue(100, 'float')                                   # FIFO队列，由100个元素
enqueue_op = queue.enqueue([tf.random_normal([1])])                  # 定义入队操作

qr = tf.train.QueueRunner(queue, [enqueue_op] * 5)                   # 创建多个线程的入队操作，QueueRunner的第一个参数为被操作队列，第二个为启动五个线程
tf.train.add_queue_runner(qr)                                        # 将QueueRunner加入TensorFLow指定的集合中，没有指定集合则加入默认集合tf.GraphKeys.QUEUE_RUNNERS集合
out_tensor = queue.dequeue()                                         # 定义出队操作

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)   # 定义的线程在哪个会话sess，受哪个Coordinator控制

    for _ in range(3):
        print(sess.run(out_tensor)[0])

    coord.request_stop()
    coord.join(threads)