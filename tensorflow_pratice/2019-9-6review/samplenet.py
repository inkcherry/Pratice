from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


learning_rate=0.2
train_epoch=10
batch_size = 100
display_epoch=1




hidden1=256
hidden2=256

inputs=784
classes=10

x=tf.placeholder(tf.float32,[None,inputs])
y=tf.placeholder(tf.float32,[None,classes])



layer_hidden1=tf.Variable(tf.random_normal([inputs,hidden1]))
layer_hidden2=tf.Variable(tf.random_normal([hidden1,hidden2]))
layer_ouput=tf.Variable(tf.random_normal([hidden2,classes]))



biases_of_hidden1=tf.Variable(tf.random_normal(hidden1))
biases_of_hidden2=tf.Variable(tf.random_normal(hidden2))
biases_of_out=tf.Variable(tf.random_normal(classes))

def nn(input_):
    layer1=tf.add (tf.matmul(input_,layer_hidden1),biases_of_hidden1)
    layer2=tf.add(tf.matmul(layer1,layer_hidden2),biases_of_hidden2)
    out_layer = tf.add(tf.matmul(layer2,layer_ouput),biases_of_out)
    return out_layer
    
    
logits=nn(x) 
pre = tf.softmax(logits)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))

    
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

accuracy =tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pre),tf.argmax(y)),tf.float32))


init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(train_epoch):
        
        batchs = mnist.train.num_examples/batch_size

        for i in range(batchs):
            cur_x,cur_y =mnist.train.next_batch(batch_size)
            _,cur_loss=sess.run([optimizer,loss],feed_dict={x:cur_x,y:cur_y})
        
        if epoch%display_epoch==0:
            print('epoch=',epoch,',cur_loss=',loss)
        print('acc=',accuracy.eval({x:cur_x,y:cur_y}))
        
    print('finish')
    
        
            
            

    

    




