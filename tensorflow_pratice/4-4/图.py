import tensorflow as tf
#创建一个常量op  行向量
m1 = tf.constant([[3,3]])

#列向量
m2=tf.constant([[2],[3]])

product = tf.matmul(m1,m2)

print(product)

sess =tf.Session()

print(result)
sess.close()


#with tf.Session() as sess:
   # result =sess.run(product)
#print(result);
#sess.close()