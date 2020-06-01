import tensorflow as tf
import numpy as np
import pandas as pd
'''
Header=""
for i in range(1,10001):
    Header+=str(i)
    if i<10000:Header+=","
print(Header)
X=np.load('X_train/tmpx.npy')
X=X[:1000,:]
np.savetxt("tmpx.csv",X,delimiter=",",header=Header,comments="")
Y=np.load('Y_train/tmpy.npy')
Y=Y[:1000,:]
np.savetxt("tmpy.csv",Y,delimiter=",",header="1,2,3",comments="")
'''
X=pd.read_csv("tmpx.csv")
Y=pd.read_csv("tmpy.csv")

Features=[str(i) for i in range(1,10001)]
features_col=[tf.feature_column.numeric_column(k) for k in Features]

estimator=tf.estimator.LinearRegressor(
       feature_columns=features_col,
       model_dir="train"
        )
Ein=tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: X[k].values for k in Features}),
        y=pd.Series(Y['1'].values),
        batch_size=128,
        shuffle=False)
estimator.train(input_fn=Ein,steps=100)

