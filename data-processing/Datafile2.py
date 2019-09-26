# import numpy as np
# import tensorflow as tf
#
# train_x,lable=(np.random.sample([6,4,4]),np.random.sample([6,1]))
# # print(x)
# # print(y)
# data=tf.data.Dataset.from_tensor_slices((train_x,lable))
# print(data)
# data=data.shuffle(2).repeat(2).batch(2)
# train_data=data.prefetch(3)  #juset prefetch
#
#
# # NOTE: The following examples use `{ ... }` to represent the
# # contents of a dataset.
a = { 1, 2, 3 }
b = { (7, 8), (9, 10) }

# The nested structure of the `datasets` argument determines the
# structure of elements in the resulting dataset.
print(a.enumerate(start=5))
print(b.enumerate())
