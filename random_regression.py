from sklearn.ensemble import RandomForestRegressor

def NAE(y_predict,y):
    total=len(y)
    err=1./total
    #print((np.absolute(np.subtract(y_predict,y))))
    #print(np.divide(np.absolute(np.subtract(y_predict,y)),y))
    tmp=np.sum(np.divide(np.absolute(np.subtract(y_predict,y)),y))
def WMAE(y_predict,y):
    total=len(y)
    err=1./total
    multi=[[300,1,200]]*total
    multi=np.array(multi)
    print(multi)
    tmp=np.sum(np.multiply(np.absolute(np.subtract(y_predict,y)),multi))
    return err*tmp
x=np.load('X_train/arr_0.npy')
y=np.load('Y_train/arr_0.npy')
total=len(y)

print("loading done")

x_test=np.float32(x[-10:])
y_test=np.float32(y[-10:])
