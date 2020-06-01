import numpy as np
import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import matplotlib.pyplot as plt
training_step=sys.argv[0]
data_x=np.load('X_train/tmpx.npy')
data_y=np.load('Y_train/tmpy.npy')
data_y=data_y[:,0]
#print(data_y.shape)

#np.save('X_train/tmpx.npy',data_x[:1000,:])
#np.save('Y_train/tmpy.npy',data_y[:1000,:])
print("loading done")
#learning_rate=0.01
#print(data_x.shape[1:])
#print(data_y.shape)

feature_column=[tf.feature_column.numeric_column('x',shape=data_x.shape[1])]

print(feature_column)
#Optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.001)
#Optimizer=tf.contrib.estimator.clip_gradients_by_norm(Optimizer,5.0)
estimator=tf.estimator.LinearRegressor(
        feature_columns=feature_column,
 #       optimizer=Optimizer,
       # model_dir="training"
        )
estimation_input=tf.estimator.inputs.numpy_input_fn(
        x=data_x,
        y=data_y,
        #batch_size=128,
        shuffle=False,
        num_epochs=None)
estimator.train(input_fn=estimation_input,steps=100)

