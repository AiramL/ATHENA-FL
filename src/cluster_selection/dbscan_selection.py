# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python dbscan-selection.py <read_file> <clusters-size>


from pickle import load, dump
from sys import argv
from sklearn.cluster import DBSCAN
from os import listdir
from tensorflow.keras import models

# check if the parameters are correct
if not argv[1]:
    print("missing read file");

dataList = [];
clientList = [];

# receives an ordenated list of client data vectors
#for clientFile in range(1,11):
for clientFile in listdir(argv[1]+'/'):
    dataList.append(models.load_model(argv[1]+'/'+clientFile).get_weights()[-1]);
    clientList.append(clientFile.split('_')[9])

# verifying the DBSCAN hyperparameter
if argv[2] and argv[2].isnumeric:
    clientCluster = DBSCAN(eps=float(argv[2]),min_samples=2).fit(dataList);
else:
    clientCluster = DBSCAN(eps=5,min_samples=2).fit(dataList);
    

#print(clientCluster.labels_)
#print(clientList)
clients_dictionary = dict(zip(clientList,clientCluster.labels_))

print(clients_dictionary)

with open('server_connection_dictionary','wb') as writer:
    dump(clients_dictionary,writer)

