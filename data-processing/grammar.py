import numpy as np
import tensorflow as tf
print (tf.__version__)

d = [3,3,4]
a = np.zeros((2,2,2),float)
c=np.zeros(d,float)

e=np.zeros(d+[1]*d[-1])
f=np.zeros(d+[1*d[-1]])
g=np.zeros(d[:-1]+[4])

print (tf.shape(a))
print (tf.shape(c))
print (tf.shape(e))
print(tf.shape(f))
print(tf.shape(g))

