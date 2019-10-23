import numpy as np
import tensorflow as tf
import  random
from tensorflow import  keras
print (tf.__version__)
#pratice of  tsv2
#  train a function:   f(x[]) =1/(x[1]+x[2])
#  train a function:   f(x[]) =x[1]
optimizer = tf.optimizers.SGD(0.025)
batch_size=5
noise_dim=3
batchs=10000/50
training_steps = 10000
log_steps=500

# train_x=np.random.sample([10000,2])
train_x=np.zeros([10000,3],np.float32)
for i in range (10000):
    # train_x[i][0] = np.random.randint(0, 3, 6, 9)
    train_x[i][0]= (random.sample(range(0, 4), 1))[0]*3
    train_x[i][1]=3*(train_x[i][0])+10
    train_x[i][2]=train_x[i][0]+train_x[i][1]





train_data =tf.data.Dataset.from_tensor_slices((train_x))


train_data = train_data.repeat().shuffle(20).batch(batch_size).prefetch(1)

class Dis(keras.Model):
    def __init__(self):
        super(Dis,self).__init__()


        self.layer1=keras.layers.Dense(64)
        self.layer2=keras.layers.Dense(512,activation='relu')
        self.layer3=keras.layers.Dense(256,activation='relu')
        self.layer4=keras.layers.Dense(2,activation='sigmoid')   #Dense1 会报错range错误 不能跟1 求交叉熵。


    def call(self,input,training=False):
        x=self.layer1(input)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # x=tf.reduce_mean(x)   #？？？？
        return x

class Gen(keras.Model):
    def __init__(self):
        super(Gen,self).__init__()
        self.layer1 = keras.layers.Dense(batch_size*3)
        self.layer2 = keras.layers.Dense(512, activation='relu')
        self.layer3 = keras.layers.Dense(256, activation='relu')
        self.layer4 =keras.layers.Dense(3,activation='relu')

    def call(self,noise,is_training=False):
        x = tf.reshape(noise, [-1, 3])   #可以不要，noise dim 定义过了
        # print("noise shape")
        # print(tf.shape(noise))
        x=self.layer1(noise)
        # print(tf.shape(x))

        # print("after reshape")
        # print(tf.shape(x))

        x=self.layer2(x)
        # print(tf.shape(x))
        x=self.layer3(x)
        # print(tf.shape(x))
        x=self.layer4(x)
        # print(tf.shape(x))
        # print("finish call")
        return x





def get_g_loss(fake_res):
    # print("fake res")
    #
    # print(fake_res)

    g_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_res,labels=tf.ones([batch_size],dtype=tf.int32)))
    # print("reconstructed_image")
    # print(tf.shape(fake_res))
    # print("tfones")
    # print(tf.shape(tf.ones([batch_size])))
    # print("gen_loss")
    # print(tf.shape(g_loss))
    # print(g_loss)
    # g_loss=float(g_loss)
    return g_loss

def get_d_loss(fake_res,real_res):
    # return tf.reduce_mean(tf.square(y-pred))
    # print("real and re image")
    # print(tf.shape(real_res))
    # print(tf.shape(fake_res))
    # print("_________")
    d_fake_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_res,labels=tf.zeros([batch_size],dtype=tf.int32)))


    # exit()
    d_real_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_res,labels=tf.ones([batch_size],dtype=tf.int32)))
    # test
    # d_real_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_res,labels=tf.ones([batch_size],dtype=tf.int32)))

    d_loss=d_fake_loss+d_real_loss

    return d_loss





#学习率不能过低！0.0

gen = Gen()
dis = Dis()
def train_step(train_x):

    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)

    #trarin D
    d_loss=0
    g_loss=0
    with tf.GradientTape() as tape:

        fake_data=gen(noise,is_training=True)
        real_data=train_x
        # print("the input shape")
        # print(tf.shape(fake_data))
        # print(tf.shape(real_data))
        # print("______||||___")
        real_res=dis(real_data)

        fake_res=dis(fake_data)

        d_loss=get_d_loss(fake_res,real_res)


        grads=tape.gradient(d_loss,dis.trainable_variables)
        optimizer.apply_gradients(zip(grads,dis.trainable_variables))
    #train G
    with tf.GradientTape() as tape:
        fake_data = gen(noise, is_training=True)

        fake_res = dis(fake_data)

        g_loss = get_g_loss(fake_res)
        grads = tape.gradient(g_loss, gen.trainable_variables)
        optimizer.apply_gradients(zip(grads, gen.trainable_variables))

    return g_loss,d_loss



print ("_____________________________________________________________")
print ("strat training")

for step, (batch_x) in enumerate(train_data.take(training_steps), 1):
    g_loss,d_loss=train_step(batch_x)
    if step%log_steps==0:
        print("g_loss,d_loss="+str(g_loss)+" "+str(d_loss))
        # print("loss="+str(get_loss(pred,batch_y)))


# test_x=np.random.sample([50,2])
# print ("test----------------x-------")
# print(test_x[1])
# print(test_x[2])
# print(test_x[3])


test_nosie=    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)


gen_x=gen(test_nosie)
print ("pred----------------res------")
print(tf.shape(gen_x))
print(gen_x[1])
print(gen_x[2])
print(gen_x[3])






