# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import threading
import time


"""
    Coordinator类是一个多线程协调者，其提供的几个函数：
    should_stop: 启动的线程反复询问这个函数，当返回True时，线程停止
    request_stop: 任意一个线程可以调用该函数通知其他线程退出
    join: 保证主线程在子线程完成之后才执行
"""


def  MyLoop(coord, worker_id):
    while not coord.should_stop():
        if np.random.rand() < 0.1:    # 某个线程随机值小于0.1时退出
            print("线程%d请求退出所有线程"%worker_id)
            coord.request_stop()
        else:
            print("线程%d正在老老实实工作"%worker_id)
        time.sleep(1)

coord = tf.train.Coordinator()
threads = [
    threading.Thread(target=MyLoop, args=(coord, i, )) for i in range(5)
]

for t in threads:
    t.start()

coord.join(threads)