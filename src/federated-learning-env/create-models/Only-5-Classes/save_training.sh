#!/bin/bash
#
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

# initialize the server
python3.9 server.py 100 8081 &
#python3.9 server.py 100 8082 &
#python3.9 server.py 100 8083 &
#python3.9 server.py 100 8084 &
#python3.9 server.py 100 8085 &

sleep 2

# usage of client.py: python3.9 client.py <model-type> <server-port> <client-id> <basic-neural-network-flag> <number-of-clients>

# save results for 5 clients and a robust model
for i in $(seq 5)
do
	python3.9 client.py 1 8081 $i 0 5 >> results/result-complex-model-client$i &
done


# print results on the screen for 5 clients and a robust model
#for i in $(seq 5)
#do
#	python3.9 client.py 1 8081 $i 0 5 & 
#done

# save results for 5 clients and a simple model
#for i in $(seq 5)
#do
#	python3.9 client.py 0 8085  $i 1 5 >> results/result-simple-model-5-client$i &
#	python3.9 client.py 1 8081  $i 1 5 >> results/result-simple-model-1-client$i &
#	python3.9 client.py 2 8082  $i 1 5 >> results/result-simple-model-2-client$i &
#	python3.9 client.py 3 8083  $i 1 5 >> results/result-simple-model-3-client$i &
#	python3.9 client.py 4 8084  $i 1 5 >> results/result-simple-model-4-client$i &
#done


# print results for 5 clients and a simple model
#for i in $(seq 5)
#do
#	python3.9 client.py 0 8085  $i 1 5 >> results/result-simple-model-5-client$i &
#	python3.9 client.py 1 8081  $i 1 5 >> results/result-simple-model-1-client$i &
#	python3.9 client.py 2 8082  $i 1 5 >> results/result-simple-model-2-client$i &
#	python3.9 client.py 3 8083  $i 1 5 >> results/result-simple-model-3-client$i &
#	python3.9 client.py 4 8084  $i 1 5 >> results/result-simple-model-4-client$i &
#done
