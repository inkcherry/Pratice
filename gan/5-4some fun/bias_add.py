import os
import math
import numpy as np
import tensorflow as tf


tf.reset_default_graph()




#常用来做滤波器
shape1=[1,2,3,4]

w=tf.get_variable('w',shape1, initializer=tf.truncated_normal_initializer(stddev=0.02))

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     print(sess.run(w))


#kk
b = tf.get_variable('b', [4], initializer=tf.constant_initializer(1000.0))


c = tf.reshape(tf.nn.bias_add(w, b), w.get_shape())

with tf.Session() as sess:
     sess.run(tf.global_variables_initializer())
     print(sess.run(c))
     
