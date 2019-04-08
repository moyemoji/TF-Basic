# -*- coding: utf-8 -*-
import tensorflow as tf


"""
    Tensorflow队列
"""

q = tf.FIFOQueue(2, "int32")        # 定义一个FIFO队列
init = q.enqueue_many(([0,10],))    # 插入多个元素

x = q.dequeue()                     # 第一个元素出队
y = x + 1                           # 加一
q_inc = q.enqueue([y])              # 插入一个元素

with tf.Session() as sess:
    init.run()                      # 初始化操作
    for _ in range(5):
        v, _ = sess.run([x, q_inc]) # 执行计算节点
        print(v)