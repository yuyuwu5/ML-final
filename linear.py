import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
x = np.linspace(0, 50, 50) 
y = np.linspace(0, 50, 50) 
  
# Adding noise to the random linear data 
x += np.random.uniform(-4, 4, 50) 
y += np.random.uniform(-4, 4, 50) 
  
n = len(x) 
X=tf.placeholder("float")
Y=tf.placeholder("float")

learning_rate=0.01
training_epoch=1000
W=tf.Variable(np.random.randn(),name="W")
b=tf.Variable(np.random.randn(),name="b")
y_pred=tf.add(tf.multiply(X,W),b)
cost=tf.reduce_sum(tf.pow(y_pred-Y,2)/(2*n))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epoch):
        for(_x,_y)in zip(x,y):
            sess.run(optimizer,feed_dict={X:_x,Y:_y})
        if (epoch+1)%50==0:
            c=sess.run(cost,feed_dict={X:x,Y:y})
            print("Epoch: ",epoch+1,"cost= ",c,"W= ",sess.run(W),"b= ", sess.run(b))
