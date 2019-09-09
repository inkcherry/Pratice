from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

learning_rate=0.2
train_epoch=10
batch_size = 100
display_step=1

x =tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

w = tf.Variable(tf.zeros([784,10]))   
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(x,w)+b)

loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

optimizer  =tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(train_epoch):
        batchs = int(mnist.train.num_examples/batch_size)
        for i in range(batchs):
            cur_x,cur_y = mnist.train.next_batch(batch_size)
            sdf,cur_cost=sess.run([optimizer,loss],feed_dict={x:cur_x,y:cur_y})
            
        if (epoch+1)%display_step == 0:
            print('epoch=',epoch,' loss=',cur_cost)
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    print('finish')




    
            