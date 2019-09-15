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


def nn(x):
    x= x_dict['image']
    
    layer_1=tf.layers.dense[x,hidden1]
    
    layer_2=tf.layers.dense[hidden1,hidden2]
    
    out_layer=tf.layers.dense[hidden2,classes]
    
    return out_layer

def model(features,labels,mode):
#    这里的mode是什么意思
    logits =nn(x)
    
    pred_classes= tf.nn.softmax(logits)
    
    pred_probas=tf.argmax(logits,axis=1)
    
    
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    
    optimizer=tf.train.GradientDescentOptimizer(learning_rate =learning_rate)
    
    train_op = optimizer.minimize(loss_op)
    
#    acc_op=tf.reduce_mean(tf.equal(res,labels))
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    
    estim_specs =tf.estimator.EstimatorSpec
    (
     mode=mode
     predictions=pred_classes,
     loss=loss_op,
     train_op=train_op,
     eval_metric_ops={'accuracy':acc_OP})
    )
    return estim_specs
    
    
    
    

