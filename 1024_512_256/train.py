import tensorflow as tf
import numpy as np
import os

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
X=np.load('training_data/x_train.npy')
Y=np.load('training_data/y_train.npy')
#X=np.load('training_data/x_part.npy')
#Y=np.load('training_data/y_part.npy')

X=X[:25000,:]
Y=Y[:25000,:]

x={str(i): X[:,i-1] for i in range(1,10001)}
#y={str(i): Y[:,i-10001] for i in range(10001,10004)}
full=tf.data.Dataset.from_tensor_slices((x,Y)).shuffle(4096)
train_size=int(0.8*len(Y))
train=full.take(train_size)
test=full.skip(train_size)
def input_train():
	return (train.shuffle(4096).batch(128).repeat().make_one_shot_iterator().get_next())
def input_test():
	return (test.shuffle(4096).batch(128).repeat().make_one_shot_iterator().get_next())
#full=full.shuffle()
features=[tf.feature_column.numeric_column(str(i)) for i in range(1,10001)]
estimator=tf.estimator.DNNRegressor(
	feature_columns=features,
	label_dimension=3,
	hidden_units=[1024,512,256],
	model_dir="model/Dnn_1024_train"
	)
print("training")
wrap=estimator.train(input_fn=input_train,steps=5000)

'''
print("evaluating...")
Ee=lambda:inp(xx,YY)
#result=estimator.evaluate(input_fn=Ee,steps=1)
result=estimator.evaluate(input_fn=input_test,steps=1)
for i in sorted(result):
	print("%s: %s" % (i,result[i]))
a=np.load('testing_data/x_test.npy')
#test_x={str(i): a[:,i-1] for i in range(1,10001)}
#testing=tf.data.Dataset.from_tensor_slices((test_x))
#EEE=lambda: inp(test_x,None)
#predict=estimator.predict(input_fn=EEE)
print("testing")
#predict=list(estimator.predict(input_n=lambda:testing))
#predict=numpy.array(predict)
#np.savetxt("out.csv",predict,delimiter=",")
'''
