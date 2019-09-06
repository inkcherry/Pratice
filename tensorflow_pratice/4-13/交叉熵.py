
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#one_hot=True二值化 标签

#每个批次的大小
batch_size = 100

#批次个数  这个batch_size要加上！
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
#输入 784个神经元
#28*28=784图像
x = tf.placeholder(tf.float32,[None,784])  
y = tf.placeholder(tf.float32,[None,10])


#创建神经网络  （输入层，输出层）

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

prediction =tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))


#交叉熵
loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))



 # 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
 
 
 #初始化变量
init  = tf.global_variables_initializer()
 
 
 
correct_prediction =  tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
 #返回向量中最大值位置，也就是几
 
 #求准确率 通过平局值求
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))



