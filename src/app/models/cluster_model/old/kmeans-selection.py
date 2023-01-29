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
from sklearn.cluster import KMeans


dataList = [];

# receives an ordenated list of client data vectors
for clientFile in range(1,11):
    dataList.append(load(open('../client_models/client-'+str(clientFile)+'-weights','rb'))[261]);

clientCluster = KMeans(n_clusters=5, random_state=0).fit(dataList)
    

with open("cluster_model","wb") as model:
    dump(clientCluster,model)





