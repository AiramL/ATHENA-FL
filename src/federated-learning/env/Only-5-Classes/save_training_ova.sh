#!/bin/bash
#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#

data=$1
serverPort=$2
epochs=$3
numClients=$4
ova=$5
scenario=$6
labels=$7
modelType=$8

# usage of client.py: python3.9 client.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients>


# save results a simple model
if [ $ova -eq 1 ];
then
	python3.9 server.py $epochs $serverPort &

	sleep 3

	# intialize the clients
	for i in $(seq $numClients)
	do
			python3.9 client.py $modelType $serverPort $i 1 $numClients $data $scenario $labels >> results/result-$data-simple-model-$modelType-epochs-$epochs-clients-$numClients-client$i &
	done
fi

