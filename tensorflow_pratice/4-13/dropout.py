
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
keep_prob = tf.placeholder(tf.float32);

#创建神经网络  （3层隐藏层，输出层）
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob) 

W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob) 

W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob) 

W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)
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
 
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        acc2= sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc)+",Training Accuracy"+str(acc2))



