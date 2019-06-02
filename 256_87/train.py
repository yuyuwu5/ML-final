import tensorflow as tf
import numpy as np
import os

X=np.load('training_data/x_train.npy')
Y=np.load('training_data/y_train.npy')
#X=np.load('training_data/x_part.npy')
#Y=np.load('training_data/y_part.npy')

X=X[10000:35000,:]
Y=Y[10000:35000,:]

x={str(i): X[:,i-1] for i in range(1,10001)}
full=tf.data.Dataset.from_tensor_slices((x,Y)).shuffle(4096)
train=full
#train_size=int(0.8*len(Y))
#train=full.take(train_size)
#test=full.skip(train_size)
def input_train():
	return (train.shuffle(4096).batch(128).repeat().make_one_shot_iterator().get_next())
def input_test():
	return (test.shuffle(4096).batch(128).repeat().make_one_shot_iterator().get_next())
#full=full.shuffle()
features=[tf.feature_column.numeric_column(str(i)) for i in range(1,10001)]
estimator=tf.estimator.DNNRegressor(
	feature_columns=features,
	label_dimension=3,
	hidden_units=[256,87],
	model_dir="model/Dnn_256_87_train"
	)
print("training")
wrap=estimator.train(input_fn=input_train,steps=1000)
print("done")
