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
# usage: python kmeans-selection.py <read_file_data> <number-of-clusters>


from pickle import load, dump
from sys import argv
from os import listdir
from sklearn.cluster import KMeans
from tensorflow.keras import models

# check if the parameters are correct
if not argv[1]:
    print("missing read file");

layer_index = -1
if len(argv) >= 4:
    layer_index = int(argv[3])

single_layer = True

dataList = [];
clientList = [];
# receives an ordenated list of client data vectors
#for clientFile in range(1,11):
#    dataList.append(load(open(argv[1]+'/'+'client-'+str(clientFile)+'-weights','rb'))[261]);
for clientFile in listdir(argv[1]+'/'):
    if single_layer:
        dataList.append(models.load_model(argv[1]+'/'+clientFile).get_weights()[layer_index]);
    else:
        client_weights = []
        for index in range(len(models.load_model(argv[1]+'/'+clientFile).get_weights())):
            client_weights.append(models.load_model(argv[1]+'/'+clientFile).get_weights()[index])
        
        dataList.append(client_weights)

    clientList.append(clientFile.split('_')[9]+'_'+clientFile.split('_')[-1])

# verifying the kmeans hyperparameter
if argv[2] and argv[2].isnumeric:
    clientCluster = KMeans(n_clusters=int(argv[2]), random_state=0).fit(dataList)
else:
    clientCluster = KMeans(n_clusters=5, random_state=0).fit(dataList)
    
# print the clusterization result
#print(clientCluster.labels_)
#print(clientList)

clients_dictionary = dict(zip(clientList,clientCluster.labels_))

print(clients_dictionary)

with open('server_connection_dictionary','wb') as writer:
    dump(clients_dictionary,writer)





