


# lstm autoencoder to recreate a timeseries
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from codes import ObjLoader
from codes import TrainingSequence
import open3d
import sys
import os
import pickle
import random
import glob
import re
import math
import glob
import os
from personal_parameters.myConfig import config
#from codes import ObjLoader

'''
this function prepare the data  for a lSTM network 
https://stackoverflow.com/questions/57168903/autoencoder-with-3d-convolutions-and-convolutional-lstms
voir ici .

'''

def get_dataLSTM(data_dir) :
    treefiles = glob.glob(os.path.join(data_dir, "*.ply"))

    for file in treefiles:
        pcd = ObjLoader.load_point_cloud(file)
        print (pcd.vertices)
        return (pcd.vertices)
    t=[]
    x=[]
    y=[]
    z=[]

    timestamp=0

    #for file in treefiles:
    timestamp = +0.001

            ## loading points cloud
    pcd = ObjLoader.load_point_cloud(file)
    ## get the data
    t.append(timestamp)
    x.append(pcd.vertices[j][0])
    y.append(pcd.vertices[j][1])
    z.append(pcd.vertices[j][2])
    print (pcd.vertice)
    return (pcd.vertice)


path=config.pathToLongdress2Compressed_Ply
get_dataLTSM(path)
"X=nparray[[timestamp],x,y,z] 4*len(lenfiles)*"""


def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y

"output_X[i]=[X[i]|X[i+1]|X[i+2]] 4*len(lenfiles)"



treefiles = glob.glob(os.path.join(path, "*.ply"))
y=np.zeros(len(treefiles[1]))
n_features = 4
timesteps=3
path=config.pathToLongdress2Compressed_Ply
T = (get_dataLTSM(path))

X, y = temporalize( T,y, timesteps)
X = np.array(X)
X = X.reshape(X.shape[0], timesteps, n_features)



# define model
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(timesteps))

#decoder
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()

# fit model
model.fit(X, X, epochs=20, verbose=0)
model.save("my_model")
# demonstrate reconstruction
yhat = model.predict(X, verbose=0)
print('---Predicted---')
print(np.round(yhat,3))
print('---Actual---')
print(np.round(X, 3))
'''model 1 predict a trunk after x given trunk '''


