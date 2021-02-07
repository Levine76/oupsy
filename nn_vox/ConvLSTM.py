import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pylab as plt


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


def get_dimLSTM(data_dir):
    plyfiles = glob.glob(os.path.join(data_dir, "*.ply"))
    return len(plyfiles), len(plyfiles[1])

def get_sepLSTM(data_dir):
    plyfiles = glob.glob(os.path.join(data_dir, "*.ply"))
    return int(0.8*len(plyfiles[1]))




def get_dataLSTM(data_dir):
    plyfiles = glob.glob(os.path.join(data_dir, "*.ply"))
    pcd = ObjLoader.load_point_cloud(plyfiles[1])
    X = np.zeros((nb_frame,nb_point,3),dtype=np.float)

    for file in plyfiles:
        pcd = ObjLoader.load_point_cloud(file)
        np.append(X,pcd.vertices)
    return (X.transpose())

def shape_dataLSTM(X, lookback=3):
    output_X = []
    for i in range(len(X) - lookback - 1):
        t = []
        for j in range(1, lookback + 1):
            # Gather past records upto the lookback period
            t.append(X[[(i + j + 1)], :])
        output_X.append(t)
    return output_X

n_features = 3
path=config.pathToLongdress2Compressed_Ply

nb_frame,nb_point=get_dimLSTM(path)
nb_framesep=get_sepLSTM(path)


T = (get_dataLSTM(path))

X = shape_dataLSTM(T,3)
X = np.array(X)




######################################################################################
seq = keras.Sequential(
    [
        keras.Input(
            shape=(nb_point, nb_point, nb_point, 3)
        ),  # Variable-length*sequence
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(
            filters=40, kernel_size=(3, 3), padding="same", return_sequences=True
        ),
        layers.BatchNormalization(),
        layers.Conv3D(
            filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same"#worth ?
        ),
    ]
)
seq.compile(loss="binary_crossentropy", optimizer="adadelta")#mse ?
######################################################################################

epochs = nb_frame
seq.fit(
    X,
    X,
    batch_size=10,
    epochs=epochs,
    verbose=1,
    validation_split=0.1,
)

# Start from first 7 frames
K = X[:7, ::, ::, ::]

# Predict 2 frames
for j in range(2):
    new_pos = seq.predict(K[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    K = np.concatenate((K, new), axis=0)