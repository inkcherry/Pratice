import numpy as np
import tensorflow as tf
from tensorflow import  keras
print (tf.__version__)
#pratice of  tsv2
#  train a function:   f(x[]) =1/(x[1]+x[2])
#  train a function:   f(x[]) =x[1]

train_x=np.random.sample([10000,2])
print (train_x)
y=np.zeros([10000,1],float)

for i in range (10000):
    # y[i]=1/(train_x[i][0]+train_x[i][1])
    # y[i]=train_x[i][0]+train_x[i][1]
    y[i]=train_x[i][0]+train_x[i][1]

print (y)


train_data =tf.data.Dataset.from_tensor_slices((train_x,y))


train_data = train_data.repeat().shuffle(20).batch(1).prefetch(1)

class MYModel(keras.Model):
    def __init__(self):
        super(MYModel,self).__init__()


        self.layer1=keras.layers.Dense(64)
        self.layer2=keras.layers.Dense(512,activation='relu')
        self.layer3=keras.layers.Dense(256,activation='relu')
        self.layer4=keras.layers.Dense(1,activation='relu')   #res >1


    def call(self,input,training=False):
        x=self.layer1(input)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # x=tf.reduce_mean(x)   #？？？？
        return x


def get_loss(pred,y):
    # return tf.reduce_mean(tf.square(y-pred))

    return tf.reduce_mean(tf.abs(y-pred))
    # return tf.nn.softmax_cross_entropy_with_logits(pred,y)
    # return tf.nn.sigmoid_cross_entropy_with_logits(x,y)


model=MYModel()
optimizer = tf.optimizers.SGD(0.025)
#学习率不能过低！0.0

def train_step(x,y):
    with tf.GradientTape() as tape:
        pred=model(x,training=True)
        loss=get_loss(pred,y)
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))



batchs=10000/50
training_steps = 20000
log_steps=200

print ("_____________________________________________________________")
print ("strat training")

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    train_step(batch_x,batch_y)
    if step%log_steps==0:
        pred=model(batch_x)
        print("loss="+str(get_loss(pred,batch_y)))


test_x=np.random.sample([50,2])
print ("test----------------x-------")
print(test_x[1])
print(test_x[2])
print(test_x[3])

pred_y=model(test_x)
print ("pred----------------res------")
print(tf.shape(pred_y))
print(pred_y[1])
print(pred_y[2])
print(pred_y[3])






