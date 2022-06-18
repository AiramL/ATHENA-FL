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

from sys import argv

from load_federated_data import *
from generate_neural_network import build_model


# client configuration
serverPort = '8080'
modelType = 1
clientID = 1
numClients = 10
basicNN = True
dataset_name = "CIFAR-10"
scenario = 2

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

if len(argv) >= 8:
    scenario = argv[7]

# Loading the dataset
trPer = 0.9

if scenario == 1:
    x_train, y_train, x_test, y_test = load_data_federated_IID(dataset_name, clientID, numClients, basicNN, modelType, trPer)

elif scenario == 2:
    x_train, y_train, x_test, y_test = load_data_federated_2_classes(dataset_name, clientID, numClients, basicNN, modelType, trPer)

elif scenario == 3:
    x_train, y_train, x_test, y_test = load_data_federated_5_classes(dataset_name, clientID, numClients, basicNN, modelType, trPer)

# Build neural network
model = build_model(basicNN,dataset_name)


class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5,batch_size=32,steps_per_epoch=int(len(x_train)/160))
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        model.save('models/model_class_'+str(modelType)+"_simple_"+str(basicNN)+"_client_"+str(clientID))
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())


