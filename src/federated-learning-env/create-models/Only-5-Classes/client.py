# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python client.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients>

import flwr as fl
import tensorflow as tf
import numpy as np

from pickle import load
from sys import argv
from sklearn.utils import shuffle

# client configuration
serverPort = '8080'
modelType = 1
clientID = 1
numClient = 10
basicNN = True

if len(argv) >= 2:
    modelType = int(argv[1])

if len(argv) >= 3:
    serverPort = argv[2]

if len(argv) >= 4:
    clientID = int(argv[3])

if len(argv) >= 5:
    basicNN = bool(int(argv[4]))

if len(argv) >= 6:
    numClients = int(argv[5])

# Loading the dataset

if clientID <= 5:     
    # first class
    x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')),dtype=np.float32)
    y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')),dtype=np.float32)
    x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')),dtype=np.float32)
    y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')),dtype=np.float32)
    
    x_train1 = x_train1[int(len(x_train1)/numClients*(clientID-1)):int(len(x_train1)/numClients*clientID)]
    y_train1 = y_train1[int(len(y_train1)/numClients*(clientID-1)):int(len(y_train1)/numClients*clientID)]
    x_test1 = x_test1[int(len(x_test1)/numClients*(clientID-1)):int(len(x_test1)/numClients*clientID)]
    y_test1 = y_test1[int(len(y_test1)/numClients*(clientID-1)):int(len(y_test1)/numClients*clientID)]
    
    
    # second class
    x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')),dtype=np.float32)
    y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')),dtype=np.float32)
    x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')),dtype=np.float32)
    y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')),dtype=np.float32)
    

    x_train2 = x_train2[int(len(x_train2)/numClients*(clientID-1)):int(len(x_train2)/numClients*clientID)]
    y_train2 = y_train2[int(len(y_train2)/numClients*(clientID-1)):int(len(y_train2)/numClients*clientID)]
    x_test2 = x_test2[int(len(x_test2)/numClients*(clientID-1)):int(len(x_test2)/numClients*clientID)]
    y_test2 = y_test2[int(len(y_test2)/numClients*(clientID-1)):int(len(y_test2)/numClients*clientID)]



    # third class
    
    x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')),dtype=np.float32)
    y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')),dtype=np.float32)
    x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')),dtype=np.float32)
    y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')),dtype=np.float32)
    
    x_train3 = x_train3[int(len(x_train3)/numClients*(clientID-1)):int(len(x_train3)/numClients*clientID)]
    y_train3 = y_train3[int(len(y_train3)/numClients*(clientID-1)):int(len(y_train3)/numClients*clientID)]
    x_test3 = x_test3[int(len(x_test3)/numClients*(clientID-1)):int(len(x_test3)/numClients*clientID)]
    y_test3 = y_test3[int(len(y_test3)/numClients*(clientID-1)):int(len(y_test3)/numClients*clientID)]
   
    
    # fourth class

    x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')),dtype=np.float32)
    y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')),dtype=np.float32)
    x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')),dtype=np.float32)
    y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')),dtype=np.float32)
    
    x_train4 = x_train4[int(len(x_train4)/numClients*(clientID-1)):int(len(x_train4)/numClients*clientID)]
    y_train4 = y_train4[int(len(y_train4)/numClients*(clientID-1)):int(len(y_train4)/numClients*clientID)]
    x_test4 = x_test4[int(len(x_test4)/numClients*(clientID-1)):int(len(x_test4)/numClients*clientID)]
    y_test4 = y_test4[int(len(y_test4)/numClients*(clientID-1)):int(len(y_test4)/numClients*clientID)]


    
    # fifth class
    
    x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4Train','rb')),dtype=np.float32)
    y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4TrainLabel','rb')),dtype=np.float32)
    x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4Test','rb')),dtype=np.float32)
    y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4TestLabel','rb')),dtype=np.float32)
    

    x_train5 = x_train5[int(len(x_train5)/numClients*(clientID-1)):int(len(x_train5)/numClients*clientID)]
    y_train5 = y_train5[int(len(y_train5)/numClients*(clientID-1)):int(len(y_train5)/numClients*clientID)]
    x_test5 = x_test5[int(len(x_test5)/numClients*(clientID-1)):int(len(x_test5)/numClients*clientID)]
    y_test5 = y_test5[int(len(y_test5)/numClients*(clientID-1)):int(len(y_test5)/numClients*clientID)]



else:
    # first class
    x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5Train','rb')),dtype=np.float32)
    y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class5TrainLabel','rb')),dtype=np.float32)
    x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5Test','rb')),dtype=np.float32)
    y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class5TestLabel','rb')),dtype=np.float32)
    
    
    x_train1 = x_train1[int(len(x_train1)/numClients*(clientID-1)):int(len(x_train1)/numClients*clientID)]
    y_train1 = y_train1[int(len(y_train1)/numClients*(clientID-1)):int(len(y_train1)/numClients*clientID)]
    x_test1 = x_test1[int(len(x_test1)/numClients*(clientID-1)):int(len(x_test1)/numClients*clientID)]-5
    y_test1 = y_test1[int(len(y_test1)/numClients*(clientID-1)):int(len(y_test1)/numClients*clientID)]-5
    
    # second class
    x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6Train','rb')),dtype=np.float32)
    y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class6TrainLabel','rb')),dtype=np.float32)
    x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6Test','rb')),dtype=np.float32)
    y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class6TestLabel','rb')),dtype=np.float32)
    

    x_train2 = x_train2[int(len(x_train2)/numClients*(clientID-1)):int(len(x_train2)/numClients*clientID)]
    y_train2 = y_train2[int(len(y_train2)/numClients*(clientID-1)):int(len(y_train2)/numClients*clientID)]
    x_test2 = x_test2[int(len(x_test2)/numClients*(clientID-1)):int(len(x_test2)/numClients*clientID)]-5
    y_test2 = y_test2[int(len(y_test2)/numClients*(clientID-1)):int(len(y_test2)/numClients*clientID)]-5

    # third class
    
    x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7Train','rb')),dtype=np.float32)
    y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class7TrainLabel','rb')),dtype=np.float32)
    x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7Test','rb')),dtype=np.float32)
    y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class7TestLabel','rb')),dtype=np.float32)
    
   
    x_train3 = x_train3[int(len(x_train3)/numClients*(clientID-1)):int(len(x_train3)/numClients*clientID)]
    y_train3 = y_train3[int(len(y_train3)/numClients*(clientID-1)):int(len(y_train3)/numClients*clientID)]
    x_test3 = x_test3[int(len(x_test3)/numClients*(clientID-1)):int(len(x_test3)/numClients*clientID)]-5
    y_test3 = y_test3[int(len(y_test3)/numClients*(clientID-1)):int(len(y_test3)/numClients*clientID)]-5
    
    # fourth class

    x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8Train','rb')),dtype=np.float32)
    y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class8TrainLabel','rb')),dtype=np.float32)
    x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8Test','rb')),dtype=np.float32)
    y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class8TestLabel','rb')),dtype=np.float32)
    

    x_train4 = x_train4[int(len(x_train4)/numClients*(clientID-1)):int(len(x_train4)/numClients*clientID)]
    y_train4 = y_train4[int(len(y_train4)/numClients*(clientID-1)):int(len(y_train4)/numClients*clientID)]
    x_test4 = x_test4[int(len(x_test4)/numClients*(clientID-1)):int(len(x_test4)/numClients*clientID)]-5
    y_test4 = y_test4[int(len(y_test4)/numClients*(clientID-1)):int(len(y_test4)/numClients*clientID)]-5

    
    # fifth class
    
    x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9Train','rb')),dtype=np.float32)
    y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class9TrainLabel','rb')),dtype=np.float32)
    x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9Test','rb')),dtype=np.float32)
    y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class9TestLabel','rb')),dtype=np.float32)
    
    x_train5 = x_train5[int(len(x_train5)/numClients*(clientID-1)):int(len(x_train5)/numClients*clientID)]
    y_train5 = y_train5[int(len(y_train5)/numClients*(clientID-1)):int(len(y_train5)/numClients*clientID)]
    x_test5 = x_test5[int(len(x_test5)/numClients*(clientID-1)):int(len(x_test5)/numClients*clientID)]-5
    y_test5 = y_test5[int(len(y_test5)/numClients*(clientID-1)):int(len(y_test5)/numClients*clientID)]-5



# create the training data
#x_train = np.concatenate((x_train1,x_train2,x_train3,x_train4,x_train5))/255
x_train = np.concatenate((x_train1,x_train2,x_train3,x_train4,x_train5,x_test1,x_test2,x_test3,x_test4,x_test5))/255


# create the test data
#x_test = np.concatenate((x_test1,x_test2,x_test3,x_test4,x_test5))/255

# Verify if we are training a robust model or OvA models
if not basicNN:
    #y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5))
    y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5,y_test1,y_test2,y_test3,y_test4,y_test5))
    #y_test = np.concatenate((y_test1,y_test2,y_test3,y_test4,y_test5))
    
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=5, weights=None)
    model = tf.keras.applications.MobileNet((32, 32, 3), classes=5, weights=None,dropout=0.01)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# If it is a basic NN we train a One-versus-All models
else:
    # Determine what OvA model to train
    if modelType == 0:
        y_train = np.concatenate((np.ones(len(y_train1)),np.zeros(len(y_train2)*4)))
        y_test = np.concatenate((np.ones(len(y_test1)),np.zeros(len(y_test2)*4)))
    
    elif modelType == 1:
        y_train = np.concatenate((np.ones(len(y_train2)),np.zeros(len(y_train1)*4)))
        y_test = np.concatenate((np.ones(len(y_test2)),np.zeros(len(y_test1)*4)))
    
    elif modelType == 2:
        y_train = np.concatenate((np.ones(len(y_train3)),np.zeros(len(y_train1)*4)))
        y_test = np.concatenate((np.ones(len(y_test3)),np.zeros(len(y_test1)*4)))
    
    elif modelType == 3:
        y_train = np.concatenate((np.ones(len(y_train4)),np.zeros(len(y_train1)*4)))
        y_test = np.concatenate((np.ones(len(y_test4)),np.zeros(len(y_test1)*4)))
    
    elif modelType == 4:
        y_train = np.concatenate((np.ones(len(y_train5)),np.zeros(len(y_train1)*4)))
        y_test = np.concatenate((np.ones(len(y_test5)),np.zeros(len(y_test1)*4)))



    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(32, 32, 3),padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


x_train, y_train = shuffle(x_train, y_train, random_state=47527)

trSize = int(len(x_train)*0.9)

class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train[:trSize], y_train[:trSize], epochs=5,batch_size=32,steps_per_epoch=trSize/256)
        return model.get_weights(), len(x_train[:trSize]), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        if clientID == 1:
            model.save('model_class_'+str(modelType)+"_simple_"+str(basicNN))
        loss, accuracy = model.evaluate(x_train[trSize:],  y_train[trSize:], verbose=2)
        return loss, len(x_train[trSize:]), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())


