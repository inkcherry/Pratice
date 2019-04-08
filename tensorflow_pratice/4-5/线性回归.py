import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt




#使用numpy生成200个随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data =np.square(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层
#十个神经元
Weights_L1 =tf.Variable(tf.random.normal[1,10])
biases_L1 = tf.Variable(tf.zerof[1,10])
Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+biases_L1
#中间层输出  
L1 = tf.nn.tanh(Wx_plus_b_L1)

#定义神经网络输出层  一个神经元
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zerof[1,1])