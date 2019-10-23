#use zztrain gan's train_x
import numpy as np
import tensorflow as tf
import  random
from tensorflow import  keras
print (tf.__version__)
#pratice of  tsv2
#  train a function:   f(x[]) =1/(x[1]+x[2])
#  train a function:   f(x[]) =x[1]
optimizer = tf.optimizers.SGD(0.005)
batch_size=5
batchs=10000/50
training_steps = 50000
log_steps=500

# train_x=np.random.sample([10000,2])
train_x=np.zeros([10000,3],np.float32)
for i in range (10000):
    # train_x[i][0] = np.random.randint(0, 3, 6, 9)
    train_x[i][0]= (random.sample(range(0, 4), 1))[0]*3
    train_x[i][1]=3*(train_x[i][0])+10
    train_x[i][2]=train_x[i][0]+train_x[i][1]


featurs=4
hidden_1=2
hidden_2=1

class Encoder(keras.Model):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1=tf.keras.dense(hidden_2)
        self.layer2=tf.keras.dense()
    def __call__(self,input,is_train=False):
        x=self.layer1(input)
        x=self.layer2(x)
        return x

class Decoder(keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layer1=tf.keras.dense()
        self.layer2=tf.keras.dense()
    def __call__(self,input,is_training=False):
        x=self.layer1(input)
        x=self.layer2(x)
        return x




train_data =tf.data.Dataset.from_tensor_slices((train_x))




def train_step(x,y):
    with tf.GradientTape() as tape:
        pred=model(x,training=True)
        loss=get_loss(pred,y)
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))



for step, (batch_x) in enumerate(train_data.take(training_steps), 1):
    g_loss,d_loss=train_step(batch_x)
    if step%log_steps==0:
        print("g_loss,d_loss="+str(g_loss)+" "+str(d_loss))