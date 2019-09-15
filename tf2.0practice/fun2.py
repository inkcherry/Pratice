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

#shuffle打乱的混乱程度
#batch从数据集中取出数据集的个数
# prefetch指定数据集重复的次数

