
import numpy as np


def binary_labels(y_train,modelType):

    y_train_binary = []

    for label in y_train:
        if int(label) == int(modelType):
            y_train_binary.append(1)
        else:
            y_train_binary.append(0)
    
    return np.asarray(y_train_binary,dtype=np.float32)
