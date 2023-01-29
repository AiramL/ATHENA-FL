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


dataList = [];

# receives an ordenated list of client data vectors
for clientFile in range(1,11):
    dataList.append(load(open('../client_models/'+'client-'+str(clientFile)+'-weights','rb'))[261]);

# verifying the DBSCAN hyperparameter
clientCluster = DBSCAN(eps=0.0279,min_samples=2).fit(dataList);
    

with open("cluster_model","wb") as model:
    dump(clientCluster,model)



