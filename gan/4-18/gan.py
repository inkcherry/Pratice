import os
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

class GanMnist:
    @staticmethod  
    def __init__(self):
        self.mnist = input.data.read_data_sets('mnist/',one_hot=True)
        self.img_size = self.mnist.train.images[0].shape[0]
        self.batch_size = 64
        self.chunk_size=self.mnist.train.num_examples
        self.epoch_size=30
        self.sample_size=25
        
        #隐含层节点的个数
        self.units_size=128
        self.learning_rate=0.001
        #...
        self.smooth=0.1
    #G,D两个网络的定义比较简单，都是普通的nn网络    
    @staticmethod  
    def generator_n(fake_imgs,units_size,out_size,alpha=0.01):
        with tf.variable_scope('generator'):
            #定义一个全连接层
            layer1 =tf.layers.dense(fake_imgs,units_size)
            
            relu =tf.maximum(alpha*layer1,layer1)
            drop = tf.layers.dropout(relu,rete=0.2)
            
            logits =tf.layers.dense(drop,out_size)
            
            outputs =tf.tanh(logits)
            return logits,outputs
    def discriminator_n(imgs,units_size,alpha=0.01,reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            layer1=tf.layers.dense(imgs,units_size)
        
            relu=tf.maximum(alpha*layer1,layer)
            logits=tf.layers.dense(relu,1)
            outputs=tf.sigmoid(logits)
            return logits,outputs
        
        
    @staticmethod
    def loss(real_logits,fake_logits,smooth)
        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=tf.ones_like(fake_logits)*(1-smooth)))
        #si_f=sigmoid(fake_logits)
        #y=labels=ones_like(fake_logits)*(1-smooth)
        #g_loss=-(y*ln(si_f)+(1-y)*ln(1-sif))
        #generator hope fake_logits=1  ,we use (1 and fake_losgits) to get lossfun
        
        d_fake_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits,labels=tf.zeros_like(fake_logits)))
        #discriminator hope fake_logits=0  ,we use (0 and  fake_losgits) to get lossfun
        
        
        d_real_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,labels=tf.ones_like(fake_logits)*(1-smooth)))
         #discriminator hope real_logits=1  ,we use (1 and  real_losgits) to get lossfun
        
        
        d_loss=tf.add(d_fake_loss,d_real_loss)
        
        return g_loss,d_fake_loss,d_real_loss,d_loss
    
    @staticmethod
    def optimizer(g_loss,d_loss,learning_rate):
        train_vars = tf.trainable_variables()
        g_vars=[var for var in train_vars if var.name.startswith('generator')]
        d_vars=[var for var in train_vars if var.name.startswith('discriminator')]
        #这里尝试一下优化器换换
        g_optimizer =tf.train.AdamOptimizer(learning_rate).minimize(g_loss,var_list=g_vars)
        d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(d_loss,var_list=d_vars)
        return g_optimizer,d_optimizer
    
    def train(self):
        real_imgs=tf.placeholder(tf.float32,[None,self.img_size],name='real_imgs')
        fake_imgs=tf.placeholder(tf.float32,[None,self.img_size],name='fake_imgs')
        
        #g生成器
        g_logits,g_outputs=self.generator_n(fake_imgs,slef.units_size,self.img_size)
        
        
        #d判别真实图片的结果
        real_logits,real_outputs=self.discriminator_n(real_imgs,self.units_size)
        
        
        #d判别生成图片的结果
        fake_logits,fake_outputs=self.discriminator_n(g_logits,self.units_size,reuse=True)
        
        
        
        #损失
        fake_loss,fake_outputs=self.discriminator_n(gen_outputs,slef.units_size,self.img_size),reuse=True)
    
        g_loss,d_fake_loss,d_r_loss,d_loss=self.loss(real_logits,fake_logits,self.smooth)
        
        #优化器
        g_optimizer,d_optimizer=self.optimizer(g_loss,d_loss,self.learning_rate)
        
        #训练
        saver=tf.tarin.Saver()
        step=0
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epoch_size):
                for ic in range(self.chunk_size):
                    batch_imgs,ic=self.mnist.train.next_batch(self.batch_size)
                    batch_imgs=batch_imgs*2-1
                    noise_imgs = np.random.uniform(-1,1,size=(self.batch_size,self.img_size))
                    
                    ic=sess.run(g_optimizer,feed_dict={fake_imgs:noise_imgs})
                    ic=sess.run(d_optimizer,feed_dict={real_imgs:batch_imgs,fake_imgs:noise_imgs})
                    step+=1
                    
                    
                    loss_d=sess.run(d_loss,feed_dict={real_imgs:batch_imgs,fake_imgs:noise_imgs})
                    loss_real=sess.run(real_loss,feed_dict={real_imgs:batch_imgs,fake_imgs:noise_imgs})
                    loss_fake=sess.run(fake_loss,feed_dict={real_imgs:batch_imgs,fake_imgs:noise_imgs})
                    loss_g=sess.run(g_loss,feed_dict={fake_imgs:noise_imgs})
                    print(datetime.now().strftime('%c'), ' epoch:', epoch, ' step:', step, ' loss_dis:', loss_dis,
                      ' loss_real:', loss_real, ' loss_fake:', loss_fake, ' loss_gen:', loss_gen)
                    model_path = os.getcwd() + os.sep + "mnist.model"
                    saver.save(sess, model_path, global_step=step)
           
    @staticmethod         
    def gen(self):
        # 生成图片
        sample_imgs = tf.placeholder(tf.float32, [None, self.img_size], name='sample_imgs')
        g_logits, g_outputs = self.generator_n(sample_imgs, self.units_size, self.img_size)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            sample_noise = np.random.uniform(-1, 1, size=(self.sample_size, self.img_size))
            samples = sess.run(g_outputs, feed_dict={sample_imgs: sample_noise})
        with open('samples.pkl', 'wb') as f:
            pickle.dump(samples, f)

    @staticmethod
    def show():
        # 展示图片
        with open('samples.pkl', 'rb') as f:
            samples = pickle.load(f)
        fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
        for ax, img in zip(axes.flatten(), samples):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        plt.show()

        
        
            