from __future__ import print_function
import numpy as np
import tensorflow as tf


#----------------------NO.1constant----------------------------------
#a=tf.constant(2)
#b=tf.constant(3)
#d=a*b
#print(d)
#a=tf.constant([[2.,3.],[3.,4.]],dtype=tf.float32)
#b=tf.constant([[4.,5.],[1.,1.]],dtype=tf.float32)
#c=tf.matmul(a,b)
#print(c)
#for i in range(a.shape[0]):
#    for j in range(a.shape[1]):
#        print(a[i][j])


#------------------NO.2sess---------------------------------------

#a=tf.constant(2,dtype=tf.float32)
#b=tf.constant(3,dtype=tf.float32)
#
#with tf.Session() as sess:
#    print(a,b)
#    print(sess.run(a+b))
#    print(sess.run(a*b))

#-------------------

#a=tf.constant([[2.,3.]],dtype=tf.float32)
#b=tf.constant([[2],[3]],dtype=tf.float32)
#res=tf.matmul(a,b)
#with tf.Session() as sess:
#    sess.run(res)
#    print(res)

#--------------------
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
add=tf.add(a,b)
mul=tf.multiply(a,b)
with tf.Session() as sess:
    for i in range(10):
        print(sess.run(add,feed_dict={a: 5,b: i}))
        sess.run(mul,feed_dict={a: 5,b: i})
        print(mul)
#---------------------------------------------------




