import numpy as np
import tensorflow as tf
import  random
from tensorflow import  keras
print (tf.__version__)
#pratice of  tsv2

train_x=np.zeros([2,3],int)
print(tf.shape(train_x))
print (train_x)
for i in range(10):
    a= (random.sample(range(0, 4), 1))[0]*3
    print(a)
print(train_x[0][0])

# for i in range (10000):
#     # train_x[i][0] = np.random.randint(0, 3, 6, 9)
#
#     train_x = random.sample(range(0, 3), 1)*3
#     train_x[i][1]=3*(train_x[i][0])+10
#     train_x[i][2]=train_x[i][0]+train_x[i][1]
