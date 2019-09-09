from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


learning_rate=0.1
train_epoch=100
batch_size = 128
display_epoch=1


num_steps = 500
display_step=100


hidden1=256
hidden2=256

inputs=784
classes=10

x=tf.placeholder(tf.float32,[None,inputs])
y=tf.placeholder(tf.float32,[None,classes])



layer_hidden1=tf.Variable(tf.random_normal([inputs,hidden1]))
layer_hidden2=tf.Variable(tf.random_normal([hidden1,hidden2]))
layer_ouput=tf.Variable(tf.random_normal([hidden2,classes]))



biases_of_hidden1=tf.Variable(tf.random_normal([hidden1]))
biases_of_hidden2=tf.Variable(tf.random_normal([hidden2]))
biases_of_out=tf.Variable(tf.random_normal([classes]))

def nn(input_):
    layer1=tf.add (tf.matmul(input_,layer_hidden1),biases_of_hidden1)
    layer2=tf.add(tf.matmul(layer1,layer_hidden2),biases_of_hidden2)
#        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    out_layer = tf.matmul(layer2,layer_ouput)+biases_of_out
    return out_layer
    
    
logits=nn(x) 
pre = tf.nn.softmax(logits)
#loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op=optimizer.minimize(loss_op)


correct_pred = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#accuracy =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre),tf.argmax(y)),tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:   
    counter=0
    sess.run(init)
    batchs=int(mnist.train.num_examples/batch_size)
    for epoch in range(train_epoch):
        for i in range(batchs):
            counter+=1
            cur_x,cur_y =mnist.train.next_batch(batch_size)
            sess.run(train_op,feed_dict={x:cur_x,y:cur_y})  
            if counter%display_step==0 or counter ==1:
                cur_loss,acc=sess.run([loss_op,accuracy],feed_dict={x:cur_x,y:cur_y})
                print(',cur_loss=',cur_loss,'acc=',acc)    
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images,y: mnist.test.labels}))

   