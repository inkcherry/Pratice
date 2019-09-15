from __future__ import absolute_import,division,print_function
from tensorflow.keras import Model,layers
from tensorflow.keras.datasets import mnist

import tensorflow as tf
import numpy as np

# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).
num_features = 784 # data features (img shape: 28*28).


# Training parameters.
learning_rate = 0.1
training_steps = 2000
batch_size = 256
display_step = 100


# Network parameters.
n_hidden_1 = 128 # 1st layer number of neurons.
n_hidden_2 = 256 # 2nd layer number of neurons.


(x_train, y_train), (x_test, y_test) = mnist.load_data()


print(x_train.shape)

# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten images to 1-D vector of 784 features (28*28).
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# Normalize images value from [0, 255] to [0, 1].
x_train, x_test = x_train / 255., x_test / 255

train_data =tf.data.Dataset.from_tensor_slices((x_train, y_train))
print(x_train.shape)

train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
#shuffle避免过拟合

class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1=layers.Dense(n_hidden_1,activation=tf.nn.relu)
        self.fc2=layers.Dense(n_hidden_2,activation=tf.nn.relu)
        self.out=layers.Dense(num_classes,activation=tf.nn.softmax)
    def call(self,x,is_training=False):
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.out(x)
        if not is_training:
            x=tf.nn.softmax(x)
        return x
neural_net=NeuralNet()
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)

def accuracy(y_pred,y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)




optimizer = tf.optimizers.SGD(learning_rate)

def run_optimizer(x,y):
    with tf.GradientTape() as g:
        pred = neural_net(x,is_training=True)
        
        loss = cross_entropy_loss(pred,y)
        
    trainable_variables =neural_net.trainable_variables
        
    gradients=g.gradient(loss,trainable_variables)
    optimizer.apply_gradients(zip(gradients,trainable_variables))


for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    run_optimizer(batch_x,batch_y)
    if step%display_step==0:
        pred=neural_net(batch_x,is_training=True)
        loss=cross_entropy_loss(pred,batch_y)
        acc=accuracy(pred,batch_y)
        print("step:%i ,loss:%f acc:%f" %(step,loss,acc))

        
        


        



    

        
