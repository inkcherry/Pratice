from __future__ import print_function
import numpy as np
import tensorflow as tf

#just a simulation模拟预测
pre  = np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.3,0.8,0.9]])

truedata = np.array([[0,0,1],[0,0,1],[0,1,0]])

correctpre = tf.equal(tf.argmax(pre,1),tf.argmax(truedata,1))

accuracy = tf.reduce_mean(tf.cast(correctpre,tf.float32))
with tf.Session() as sess:
    print(accuracy.eval())

