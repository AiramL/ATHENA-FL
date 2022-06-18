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
#import tensorflow as tf
#import numpy as np

#from pickle import load
from sys import argv
#from sklearn.utils import shuffle

from load_federated_data import load_data_federated
from generate_neural_network import build_model

# client configuration
serverPort = '8080'
modelType = 1
clientID = 1
numClient = 10
basicNN = True
dataset_name = "CIFAR-10"

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

if len(argv) >= 7:
    dataset_name = argv[6]

# Loading the dataset
trPer = 0.9

x_train, y_train, x_test, y_test = load_data_federated(dataset_name, clientID, numClients, basicNN, modelType, trPer)

# Build neural network
model = build_model(basicNN,dataset_name)


class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5,batch_size=32)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        model.save('models/model_class_'+str(modelType)+"_simple_"+str(basicNN)+"client-"+str(clientID))
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())


