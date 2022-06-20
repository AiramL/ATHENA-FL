#!/bin/bash


data=$1
serverPort=$2
epochs=$3
numClients=$4
ova=$5
scenario=$6

# usage of save_training.sh: ./save_training.sh <dataset_name> <server_port> <number_of_global_epochs> <number_of_clients> <OvA_flag> <data_distribution_scenario>

./save_training.sh FMNIST 8081 400 100 0 1
./save_training.sh MNIST 8082 400 100 0 1
./save_training.sh CIFAR-10 8083 400 100 0 1
