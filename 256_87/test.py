import tensorflow as tf
import numpy as np
features=[tf.feature_column.numeric_column(str(i))for i in range(1,10001)]
estimator=tf.estimator.DNNRegressor(
	feature_columns=features,
	label_dimension=3,
	hidden_units=[256,87],
	model_dir="model/Dnn_256_87_train")
a=np.load('testing_data/x_test.npy')
test={str(i):a[:,i-1]for i in range(1,10001)}
testing=tf.data.Dataset.from_tensor_slices((test))
def input_test():
	return(testing.batch(128).make_one_shot_iterator().get_next())
print("testing")
predict=list(estimator.predict(input_fn=input_test))
result=[]
for item in predict:
	if result==[]:
		result=np.array(item['predictions'])
	else:
		result=np.append(result,np.array(item['predictions']),axis=0)
result=np.reshape(result,(-1,3))
np.savetxt("out_256_87.csv",result,delimiter=",")