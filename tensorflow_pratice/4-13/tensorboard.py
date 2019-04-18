# coding=utf-8
'''
tensorborad图示化网络的执行流程
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
#1.加载数据
mnist = input_data.read_data_sets('data/MNIST_data',one_hot=True)
#2.使用小批量训练，定义批次大小
batch_size = 100
#计算训练次数
batch_num = mnist.train.num_examples//batch_size
#3.定义占位符
#创建命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,[None,784],name='x_input')#输入的占位符
    y = tf.placeholder(tf.float32,[None,10],name='y_input')#输出的占位符
#4.定义网络结构
with tf.name_scope('layer'):
    with tf.name_scope('weights'):
        W = tf.Variable(tf.zeros([784,10]),name='W')
    with tf.name_scope('biase'):
        b = tf.Variable(tf.zeros([10]),name='b')
    #预测
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        predict = tf.nn.sigmoid(wx_plus_b,name='softmax')
#代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y-predict))
#梯度下降
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#5.开始训练
init = tf.global_variables_initializer()
#结果存储在布尔列表中(概率最大的位置是否相等)
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(predict,1))
#准确率(转为浮点后的均值)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(10):
        for batch in range(batch_size):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print('Iter'+str(epoch)+',Test accuracy'+str(acc))
    writer.close()
