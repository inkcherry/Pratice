import numpy as np
import tensorflow as tf
from tensorflow import  keras
print (tf.__version__)
#pratice of  tsv2
#  train a function:   f(x[]) =1/(x[1]+x[2])
#  train a function:   f(x[]) =x[1]

#the following code is to fix  "could not create cudnn handle"
physical_devices = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.experimental.set_memory_growth(physical_devices[1], True)
tf.config.experimental.set_memory_growth(physical_devices[2], True)
value_per_seq=1
time_step=5
batch_size=10
lstm_num_unit=batch_size
# data_size=10000

train_x=np.random.sample([10000,5])
#不能直接生成float32的么。
train_x = tf.cast(train_x, dtype=tf.float32)

print (train_x)
y=np.zeros([10000,1],np.float32)




for i in range (10000):
    # y[i]=1/(train_x[i][0]+train_x[i][1])
    # y[i]=train_x[i][0]+train_x[i][1]
    y[i]=0.1*train_x[i][0]+0.2*train_x[i][1]+0.3*train_x[i][2]+0.4*train_x[i][3]+0.5*train_x[i][4]




train_data =tf.data.Dataset.from_tensor_slices((train_x,y))


train_data = train_data.repeat().shuffle(20).batch(batch_size).prefetch(1)


class MYModel(keras.Model):
    def __init__(self):
        super(MYModel,self).__init__()

        self.layer1=keras.layers.LSTM(lstm_num_unit)
        self.layer2=keras.layers.Dense(1,activation='relu')
        # self.layer1=keras.layers.Dense(64)
        # self.layer2=keras.layers.Dense(512,activation='relu')
        # self.layer3=keras.layers.Dense(256,activation='relu')
        # self.layer4=keras.layers.Dense(1,activation='relu')   #res >1


    def call(self,input,training=False):

        # print("the input shape")



        input =tf.cast(input,dtype=tf.float32)
        # print(tf.shape(input))
        # print(input)

        x = tf.reshape(input,[10,5,1])
        #reshape 能改变类型吗
        # print("after reshape")
        # print(x)
        # print(tf.shape(x))
        x=self.layer1(x)
        # print("after lstm")
        # print(tf.shape(x))
        # print(x)
        x=self.layer2(x)
        # x=tf.reduce_mean(x)   #？？？？
        return x


def get_loss(pred,y):
    # return tf.reduce_mean(tf.square(y-pred))

    return tf.reduce_mean(tf.abs(y-pred))
    # return tf.nn.softmax_cross_entropy_with_logits(pred,y)
    # return tf.nn.sigmoid_cross_entropy_with_logits(x,y)


model=MYModel()
optimizer = tf.optimizers.SGD(0.01)
#学习率0.025就不行

def train_step(x,y):
    with tf.GradientTape() as tape:
        pred=model(x,training=True)
        loss=get_loss(pred,y)
    grads=tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))



batchs=10000/50
training_steps = 10000
log_steps=200

print ("_____________________________________________________________")
print ("strat training")

for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
    train_step(batch_x,batch_y)
    if step%log_steps==0:
        pred=model(batch_x)
        print("loss="+str(get_loss(pred,batch_y)))


test_x=np.random.sample([batch_size,5])
print ("test----------------x-------")
print(test_x[1])
print(test_x[2])
print(test_x[3])



true_y = np.zeros([batch_size,1])
for i in range(batch_size):
    true_y[i]=0.1*test_x[i][0]+0.2*test_x[i][1]+0.3*test_x[i][2]+0.4*test_x[i][3]+0.5*test_x[i][4]

pred_y=model(test_x)
print ("the --------true_res-------real_res------")
print(tf.shape(pred_y))
print(tf.shape(true_y))
for i in range(batch_size):
    print("a pair  true and pred")
    print(true_y[i])
    print(pred_y[i])








