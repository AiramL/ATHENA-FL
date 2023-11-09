#!/bin/bash


# c = 1 (kmeans)
# c = 2 (dbscan)

kmeans=1
dbscan=2

c=$1

# p is the clustering parameter

p=$2

# l is the layer to user in the clustering algorithm

l=$3

if [ $c -eq $kmeans ]; 
then
	python3.9 ../../cluster_selection/kmeans_selection.py models_test $p $l
fi 

if [ $c -eq $dbscan ];
then
	python3.9 ../../cluster_selection/dbscan_selection.py models_test $p
fi


