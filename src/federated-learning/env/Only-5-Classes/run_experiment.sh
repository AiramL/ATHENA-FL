#!/bin/bash


data=$1
serverPort=$2
epochs=$3
numClients=$4
ova=$5
scenario=$6

# usage of save_training.sh: ./save_training.sh <dataset_name> <server_port> <number_of_global_epochs> <number_of_clients> <OvA_flag> <data_distribution_scenario>

# results_jun
#./save_training.sh fmnist 8081 400 100 0 1
#./save_training.sh mnist 8082 400 100 0 1
#./save_training.sh cifar-10 8083 400 100 0 1




# Centralized Results
#python3.9 centralized_scenario.py CIFAR-10
#python3.9 centralized_scenario.py MNIST >> results/centralized-result-MNIST-complex-model-epochs-100 & 
#python3.9 centralized_scenario.py FMNIST


# results_jul
#./save_training.sh FMNIST 8081 100 40 0 1
#./save_training.sh MNIST 8082 400 20 0 1 
#./save_training.sh CIFAR-10 8083 100 2 0 1


# results_nov
#./save_training.sh FMNIST 8081 200 40 1 2

####./save_training.sh CIFAR-10 8082 200 20 0 1 
#./save_training.sh CIFAR-10 8082 200 20 0 2 
#./save_training.sh CIFAR-10 8083 200 20 0 3 

#./save_training.sh CIFAR-10 8080 200 40 1 3

# result ova new
# <data> <serverPort> <epochs> <numClients> <ova> <scenario> <labels> <modelType>
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [2,3] 3
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [2,3] 2
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [0,1] 0
#./save_training_ova.sh CIFAR-10 8082 200 20 1 4 [0,1] 1
#./save_training_ova.sh CIFAR-10 8083 200 20 1 4 [4,5] 4
#./save_training_ova.sh CIFAR-10 8084 200 20 1 4 [4,5] 5
#./save_training_ova.sh CIFAR-10 8085 200 20 1 4 [6,7] 6
#./save_training_ova.sh CIFAR-10 8086 200 20 1 4 [6,7] 7
#./save_training_ova.sh CIFAR-10 8087 200 20 1 4 [8,9] 8
#./save_training_ova.sh CIFAR-10 8088 200 20 1 4 [8,9] 9

./save_training_ova.sh MNIST 8081 200 20 1 4 [0,1] 0
#./save_training_ova.sh MNIST 8082 200 20 1 4 [0,1] 1
#./save_training_ova.sh MNIST 8081 200 20 1 4 [2,3] 3
#./save_training_ova.sh MNIST 8081 200 20 1 4 [2,3] 2
#./save_training_ova.sh MNIST 8083 200 20 1 4 [4,5] 4
#./save_training_ova.sh MNIST 8084 200 20 1 4 [4,5] 5
#./save_training_ova.sh MNIST 8085 200 20 1 4 [6,7] 6
#./save_training_ova.sh MNIST 8086 200 20 1 4 [6,7] 7
#./save_training_ova.sh MNIST 8087 200 20 1 4 [8,9] 8
#./save_training_ova.sh MNIST 8088 200 20 1 4 [8,9] 9

#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [0,1,2,3,4] 0
#./save_training_ova.sh CIFAR-10 8082 200 20 1 4 [0,1,2,3,4] 1
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [0,1,2,3,4] 3
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [0,1,2,3,4] 2
#./save_training_ova.sh CIFAR-10 8083 200 20 1 4 [0,1,2,3,4] 4
#./save_training_ova.sh CIFAR-10 8084 200 20 1 4 [5,6,7,8,9] 5
#./save_training_ova.sh CIFAR-10 8085 200 20 1 4 [5,6,7,8,9] 6
#./save_training_ova.sh CIFAR-10 8086 200 20 1 4 [5,6,7,8,9] 7
#./save_training_ova.sh CIFAR-10 8087 200 20 1 4 [5,6,7,8,9] 8
#./save_training_ova.sh CIFAR-10 8088 200 20 1 4 [5,6,7,8,9] 9
