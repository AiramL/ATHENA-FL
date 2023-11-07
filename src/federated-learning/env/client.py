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
# usage: python client.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients> <dataset_name> <scenario> <labels>

import flwr as fl

from sys import argv
from pickle import load

from load_federated_data import *
from generate_neural_network import build_model

VERBOSE = 0

# client default configuration
serverPort = '8080'
modelType = 1
clientID = 1
numClients = 10
basicNN = True
dataset_name = "CIFAR-10"
scenario = 2
labels = [0,1]
CID=0

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
    scenario = int(argv[7])

if len(argv) >= 9:
    aux = argv[8]
    labels = []
    for item in aux:
        if item.isnumeric():
            labels.append(int(item))

if len(argv) >= 10:
    CID = int(argv[9])

    
# Loading the dataset
trPer = 0.85

if scenario == 1:
    x_train, y_train, x_test, y_test = load_data_federated_IID(dataset_name, clientID, numClients, basicNN, modelType, trPer)

elif scenario == 2:
    x_train, y_train, x_test, y_test = load_data_federated_2_classes(dataset_name, clientID, numClients, basicNN, modelType, trPer)

elif scenario == 3:
    x_train, y_train, x_test, y_test = load_data_federated_5_classes(dataset_name, clientID, numClients, basicNN, modelType, trPer)

elif scenario == 4:
    x_train, y_train, x_test, y_test = load_data_federated_by_class(dataset_name, clientID, numClients, basicNN, modelType, trPer, labels)

elif scenario == 5:
    x_train, y_train, x_test, y_test = load_dirichlet_data(dataset_name, clientID, basicNN, modelType, trPer)


if VERBOSE:
    print("Client ",clientID," Number of test samples: ",len(x_test))
    print("Client ",clientID," Number of test labels: ",len(y_test))
    print("Client ",clientID," Number of train samples: ",len(x_test))
    print("Client ",clientID," Number of train labels: ",len(y_train))

# Build neural network
model = build_model(basicNN,dataset_name,1)

# Model batch size
bs=32

class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.save('models/model_class_'+str(modelType)+"_global_before_training_simple_"+str(basicNN)+"_client_"+str(clientID)+'_dataset_'+dataset_name+'_numClients_'+str(numClients)+'_CID_'+str(CID))
        model.fit(x_train, y_train, epochs=5,batch_size=bs,steps_per_epoch=int(len(x_train)//bs))
        model.save('models/model_class_'+str(modelType)+"_local_after_train_simple_"+str(basicNN)+"_client_"+str(clientID)+'_dataset_'+dataset_name+'_numClients_'+str(numClients)+'_CID_'+str(CID))
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        model.save('models/model_class_'+str(modelType)+"_aggregated_final_simple_"+str(basicNN)+"_client_"+str(clientID)+'_dataset_'+dataset_name+'_numClients_'+str(numClients)+'_CID_'+str(CID))
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	
if basicNN:
    print('OvA Model Traning')
    # need to implement the get_cluster_id automatic
    #with open('clusters','rb') as cluster_ids:
    #    cluster_ids_list = load(cluster_ids)
        #fl.client.start_numpy_client(server_address="[::]:"+serverPort, client=CifarClient())
    # FMNIST
    #clusters = [ 0, -1,  0,  1,  2,  2,  2,  2,  2,  
    #             2, -1, -1, -1,  2,  2, -1, -1,  2,  
    #             2, -1,  3,  2,  2,  2,  2,  2, -1,  
    #             1, -1,  2,  1,  2,  2,  3,  2,  2, 
    #            -1,  1,  2,  2,  2,  2,  1,  2,  2,  
    #             3,  2, -1, -1,  2]
    with open('server_connection_dictionary','rb') as reader:
        clients_dictionary = load(reader)

    serverPort=str(int(serverPort)+clients_dictionary[str(clientID)])
    fl.client.start_numpy_client(server_address="[::]:"+serverPort, client=CifarClient())

else:
    fl.client.start_numpy_client(server_address="[::]:"+serverPort, client=CifarClient())


