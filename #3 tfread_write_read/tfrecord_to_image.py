import tensorflow as tf


reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(    # 创建一个队列来维护输入文件列表
    ["./tfrecord/output.tfrecords"]
)


_, serialized_example = reader.read(filename_queue)    # 从文件中读取一个样本，读取多个样本用read_up_to()
features = tf.parse_single_example(    # 解析读入的样本，解析多个样本用parse_example()
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    }
)

image = tf.decode_raw(features['image_raw'], tf.uint8)    # decode_raw()将字符串解析成图像对应的像素数组
label = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
coord = tf.train.Coordinator()    # 启动多线程处理输入数据
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):    # 每次运行可以读取TFRecord 文件中的一个样， 当所有样例读完之后，在此样例中程序会再从头再读
    print(sess.run([image, label, pixels]))