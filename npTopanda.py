import numpy as np
import tensorflow as tf
data_x=np.load('X_train/arr_0.npy')
data_y=np.load('Y_train/arr_0.npy')
test_x=np.load('X_test/arr_0.npy')
Header=""
for i in range(1,10001):
    Header+=str(i)
    if i<10000:Header+=","
np.savetxt("X.csv",data_x,delimiter=",",header=Header,comments="")
np.savetxt("Y.csv",data_y,delimiter=",",header="1,2,3",comments="")
np.savetxt("test_x.csv",data_x,delimiter=",",header=Header,comments="")

np.savetxt("tmpX.csv",data_x[:1000,:],delimiter=",",header=Header,comments="")
np.savetxt("tmpY.csv",data_y[:1000,:],delimiter=",",header="1,2,3",comments="")
