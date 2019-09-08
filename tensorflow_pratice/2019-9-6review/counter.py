from __future__ import print_function
import numpy as np
import tensorflow as tf

one = tf.constant(1)
a = tf.Variable(0,name="counter")
res=tf.add(a,one)
update = tf.assign(a,res)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    for _ in range(3):
        sess.run(update)
        print(sess.run(a))    