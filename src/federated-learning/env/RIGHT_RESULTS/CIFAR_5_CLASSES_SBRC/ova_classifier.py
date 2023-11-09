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
# usage: python ova-classifier.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients>

import flwr as fl
import tensorflow as tf
import numpy as np

from pickle import load, dump
from sys import argv
from sklearn.metrics import accuracy_score
from load_federated_data import *

dataset_name = 'CIFAR-10'
ova_model = {}

data=1
if len(argv) >= 2:
    data=int(argv[1])

for index in range(10):
    ova_model["model_"+str(index)] = tf.keras.models.load_model('ova_model/model_'+str(index))


# load the dataset
data_dict = {}

for index in range(1,11):
    X,Y,x,y = load_data_federated_IID(dataset_name,data,10,0,0,0.8)

# classify the samples
result = []

#for index in range(10):
#    result.append([])


#for data in range(1,11):
print('predicting dataset ',data)
for index in range(len(x)):
    partial_result = []
    for key in ova_model.keys():
        partial_result.append(ova_model[key].predict(tf.reshape(x[index],[1,32,32,3])))
    #result[data-1].append(partial_result.index(max(partial_result)))    
    result.append(partial_result.index(max(partial_result)))    

with open("result-"+str(data),"wb") as write_file:
    dump(result,write_file)

with open("y-"+str(data),"wb") as write_file:
    dump(y,write_file)

#for sample in x_test:
#    predicted_labels = []
#    for index in range(10):
#        predicted_labels.append(ova_model["model_"+str(index)].predict(tf.reshape(sample,[1,32,32,3])))
#    tmp = max(predicted_labels)
#    label = predicted.index(tmp)
#    result.append(label)

# compare the result
#accuracies = []

#for data,y_hat in enumerate(result): 
#    print('calculating accuracy dataset ',data)
#    accuracies.append(accuracy_score(data_dict['y'+str(data+1)],y_hat))


print('finished')
#print(accuracies)
#print(np.mean(accuracies))
#print(np.std(accuracies))


