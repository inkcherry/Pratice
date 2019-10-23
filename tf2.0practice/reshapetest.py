import tensorflow as tf
import numpy as np
npar1=np.zeros([50,2])
print(tf.shape(npar1))
npar2=tf.reshape(npar1,[50,2,1])

print(tf.shape(npar2))



