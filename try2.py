import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#x_data = np.linspace(-1,1,300)[:,np.newaxis]#300*1
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.square(x_data) - 0.5 + noise

#read data
x_data_all = np.load('X_train/tmpx.npy')
#x_data_all = x_data_all.f.arr_0
y_data_all = np.load('Y_train/tmpy.npy')
#y_data_all = y_data_all.f.arr_0

x_data = x_data_all[:100,:]
y_data = y_data_all[:100,:]
noise = np.random.normal(0, 0.05, y_data.shape)
y_data = y_data + noise

#read data

xs = tf.placeholder(tf.float32, [None,10000])
ys = tf.placeholder(tf.float32, [None,3])
l1 = add_layer(xs, 10000, 40, activation_function=tf.nn.tanh)
l2 = add_layer(l1, 40, 60, activation_function=tf.nn.tanh)
l3 = add_layer(l2, 60, 30, activation_function=tf.nn.tanh)
prediction = add_layer(l3, 30, 3, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.03).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
np.random.seed(10)
N = len(y_data)
print(N)
n = (int)(0.4 * N)
for i in range(1000):
    index = np.random.choice(N, n, replace = True)
    x_train = x_data
    y_train = y_data
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
    if i % 50 == 0:
        print(sess.run(loss,feed_dict={xs:x_train, ys:y_train}))
