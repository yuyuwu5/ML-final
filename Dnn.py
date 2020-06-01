import tensorflow as tf
import numpy as np
import pandas as pd
X=pd.read_csv("X.csv")
Y=pd.read_csv("Y.csv")

x_train＝X.head(X.shape[0]-300)
x_eval=X.tail(300)
y_train＝Y.head(Y.shape[0]-300)
y_eval=Y.tail(300)


Features=[str(i) for i in range(1,10001)]
features_col=[tf.feature_column.numeric_column(k) for k in Features]

def Input_fn(df,dfy,training=True):
    xx={k:tf.constant(df[k].values) for k in Features}
    if training:
        yy=tf.constant([dfy['1'].values,dfy['2'],values,dfy['3'].values])
        return xx,yy
    return xx

'''
estimator=tf.estimator.DNNRegressor(
       feature_columns=features_col,
       hidden_units=[20,30], #1024,512,256
       model_dir="model/Dnn_train"
        )
wrap=estimator.fit(input_fn=Input_fn(x_train,y_train),steps=1000)
print('Evaluating.... ')
result=estimator.evaluate(input_fn=Input_fn(x_eval,y_eval)),steps=1)
for i in sorted(result):
    print("%s: %s" %(i,result[i]))
predict=estimator.predict(input_fn=Input_fn(x_test,y_train,False))
print(predict)
out=np.array(predict)
np.savetxt("out",out,delimeter=",")
'''
