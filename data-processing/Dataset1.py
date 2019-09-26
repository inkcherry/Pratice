import numpy as np
import tensorflow as tf
print (tf.__version__)
#--------------------------------
train_x,lable=(np.random.sample([6,4,4]),np.random.sample([6,1]))
# print(x)
# print(y)
data=tf.data.Dataset.from_tensor_slices((train_x,lable))
print(data)
data=data.shuffle(2).repeat(2).batch(2)
train_data=data.prefetch(3)  #juset prefetch


for (batch_x,batch_y) in enumerate(train_data.take(5),1):
    print(batch_x)
    print(batch_y)






