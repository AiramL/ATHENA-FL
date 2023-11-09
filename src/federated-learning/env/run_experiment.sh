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


if test -d results; then
       echo "Directory in use, stopping the execution."
       exit 1
else
       mkdir results
       mkdir models
fi


# Centralized Results
#python3.9 centralized_scenario.py CIFAR-10
#python3.9 centralized_scenario.py MNIST >> results/centralized-result-MNIST-complex-model-epochs-100 & 
#python3.9 centralized_scenario.py FMNIST

# result centralizade

# initialize the server (only for FedAVG)
#python3.9 server.py 200 8080 &

# Test model
#python3.9 server.py 1 8080 1 50 50 &

# FedAVG
python3.9 server.py 200 8080 1 &

# OVA
#python3.9 server.py 200 8079 1 &
#python3.9 server.py 200 8080 1 &
#python3.9 server.py 200 8081 1 &
#python3.9 server.py 200 8082 1 &
#python3.9 server.py 200 8083 1 &
#python3.9 server.py 200 8084 1 &

#python3.9 server.py 200 8080 1 &
sleep 5

# 		    <data>   <serverPort> <epochs> <numClients> <ova>  <scenario>          <labels>        <modelType>  <CID>
#./save_training.sh CIFAR-10     8080       200         50         0       5        [0,1,2,3,4,5,6,7,8,9]       0         0
#./save_training.sh  CIFAR-10      8080         1         50         0       4        [0,1,2,3,4,5,6,7,8,9]       0         0
#./save_training.sh  CIFAR-10      8080         1         10         0       4        [0,1]       0         0
#./save_training.sh  CIFAR-10      8080         1         10         0       4        [2,3]       0         1
#./save_training.sh  CIFAR-10      8080         1         10         0       4        [4,5]       0         2
#./save_training.sh  CIFAR-10      8080         1         10         0       4        [6,7]       0         3
#./save_training.sh  CIFAR-10      8080         1         10         0       4        [8,9]       0         4
#./save_training.sh  CIFAR-10      8080         1         50         0       5        [0,1,2,3,4,5,6,7,8,9]       0         0
#./save_training.sh  FMNIST      8080         1         50         0       4        [0,1,2,3,4,5,6,7,8,9]       0         0

# OVA Dirichlet
# 		    <data>   <serverPort> <epochs> <numClients>  <ova>  <scenario>          <labels>        <modelType>  <CID>
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       0         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       1         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       2         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       3         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       4         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       5         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       6         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       7         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       8         0
#./save_training.sh   FMNIST      8080       200         50         1        5        [0,1,2,3,4,5,6,7,8,9]       9         0

# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario>          <labels>        <modelType>  <CID>
#./save_training.sh CIFAR-10     8080        200        50         0       4        [0,1,2,3,4,5,6,7,8,9]       0         1

# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario> <labels> <modelType>  <CID>
#./save_training.sh CIFAR-10     8080        200        10         0       4        [0,1]       0         1
#./save_training.sh CIFAR-10     8080        200        10         0       4        [2,3]       0         2
#./save_training.sh CIFAR-10     8080        200        10         0       4        [4,5]       0         3
#./save_training.sh CIFAR-10     8080        200        10         0       4        [6,7]       0         4
#./save_training.sh CIFAR-10     8080        200        10         0       4        [8,9]       0         5



# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario>     <labels>   <modelType>  <CID>
#./save_training.sh CIFAR-10     8080        200        25         0       4        [0,1,2,3,4]       0         1
#./save_training.sh CIFAR-10     8080        200        25         0       4        [5,6,7,8,9]       0         2


# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario>          <labels>        <modelType>  <CID>
#./save_training.sh  MNIST       8080        200        50         0       4        [0,1,2,3,4,5,6,7,8,9]      0         1


# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario>     <labels>   <modelType>  <CID>
#./save_training.sh  MNIST       8080        200        25         0       4        [0,1,2,3,4]       0         1
#./save_training.sh  MNIST       8080        200        25         0       4        [5,6,7,8,9]       0         2


# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario> <labels> <modelType>  <CID>

#./save_training.sh  MNIST       8080        200        10         0       4        [0,1]       0         1
#./save_training.sh  MNIST       8080        200        10         0       4        [2,3]       0         2
#./save_training.sh  MNIST       8080        200        10         0       4        [4,5]       0         3
#./save_training.sh  MNIST       8080        200        10         0       4        [6,7]       0         4
#./save_training.sh  MNIST       8080        200        10         0       4        [8,9]       0         5

# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario>          <labels>        <modelType>  <CID>
#./save_training.sh  FMNIST       8080        200        50         0       4        [0,1,2,3,4,5,6,7,8,9]      0         1


# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario>     <labels>   <modelType>  <CID>
./save_training.sh  FMNIST       8080        200        50         0       5        [0,1,2,3,4]       0         1
#./save_training.sh  FMNIST       8080        200        25         0       4        [5,6,7,8,9]       0         2

# 		    <data>   <serverPort> <epochs> <numClients> <ova> <scenario> <labels> <modelType>  <CID>

#./save_training.sh FMNIST       8080        200        10         0       4        [0,1]       0         1
#./save_training.sh FMNIST       8080        200        10         0       4        [2,3]       0         2
#./save_training.sh FMNIST       8080        200        10         0       4        [4,5]       0         3
#./save_training.sh FMNIST       8080        200        10         0       4        [6,7]       0         4
#./save_training.sh FMNIST       8080        200        10         0       4        [8,9]       0         5

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



# CIFAR-10 Non-IID 2 classes
#./save_training_ova.sh MNIST 8081 200 4 1 4 [0,1] 0
#./save_training_ova.sh MNIST 8082 200 4 1 4 [0,1] 1
#./save_training_ova.sh MNIST 8083 200 4 1 4 [2,3] 3
#./save_training_ova.sh MNIST 8084 200 4 1 4 [2,3] 2
#./save_training_ova.sh MNIST 8085 200 4 1 4 [4,5] 4
#./save_training_ova.sh MNIST 8086 200 4 1 4 [4,5] 5
#./save_training_ova.sh MNIST 8087 200 4 1 4 [6,7] 6
#./save_training_ova.sh MNIST 8088 200 4 1 4 [6,7] 7
#./save_training_ova.sh MNIST 8089 200 4 1 4 [8,9] 8
#./save_training_ova.sh MNIST 8080 200 4 1 4 [8,9] 9

# CIFAR-10 Non-IID 5 classes
#./save_training_ova.sh CIFAR-10 8080 200 20 1 4 [0,1,2,3,4] 0
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [0,1,2,3,4] 1
#./save_training_ova.sh CIFAR-10 8082 200 20 1 4 [0,1,2,3,4] 3
#./save_training_ova.sh CIFAR-10 8089 200 20 1 4 [0,1,2,3,4] 2
#./save_training_ova.sh CIFAR-10 8083 200 20 1 4 [0,1,2,3,4] 4
#./save_training_ova.sh CIFAR-10 8084 200 20 1 4 [5,6,7,8,9] 5
#./save_training_ova.sh CIFAR-10 8085 200 20 1 4 [5,6,7,8,9] 6
#./save_training_ova.sh CIFAR-10 8086 200 20 1 4 [5,6,7,8,9] 7
#./save_training_ova.sh CIFAR-10 8087 200 20 1 4 [5,6,7,8,9] 8
#./save_training_ova.sh CIFAR-10 8088 200 20 1 4 [5,6,7,8,9] 9


# CIFAR-10 IID
#./save_training_ova.sh CIFAR-10 8080 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 0
#./save_training_ova.sh CIFAR-10 8081 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 1
#./save_training_ova.sh CIFAR-10 8082 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 3
#./save_training_ova.sh CIFAR-10 8083 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 2
#./save_training_ova.sh CIFAR-10 8089 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 4
#./save_training_ova.sh CIFAR-10 8084 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 5
#./save_training_ova.sh CIFAR-10 8085 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 6
#./save_training_ova.sh CIFAR-10 8086 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 7
#./save_training_ova.sh CIFAR-10 8087 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 8
#./save_training_ova.sh CIFAR-10 8088 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 9

# MNIST IID
#./save_training_ova.sh MNIST 8080 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 0
#./save_training_ova.sh MNIST 8081 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 1
#./save_training_ova.sh MNIST 8082 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 3
#./save_training_ova.sh MNIST 8083 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 2
#./save_training_ova.sh MNIST 8089 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 4
#./save_training_ova.sh MNIST 8084 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 5
#./save_training_ova.sh MNIST 8085 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 6
#./save_training_ova.sh MNIST 8086 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 7
#./save_training_ova.sh MNIST 8087 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 8
#./save_training_ova.sh MNIST 8088 200 20 1 4 [0,1,2,3,4,5,6,7,8,9] 9

# MNIST Non-IID 2 classes
#./save_training_ova.sh MNIST 8080 200 20 1 4 [0,1] 0
#./save_training_ova.sh MNIST 8081 200 20 1 4 [0,1] 1
#./save_training_ova.sh MNIST 8082 200 20 1 4 [2,3] 3
#./save_training_ova.sh MNIST 8083 200 20 1 4 [2,3] 2
#./save_training_ova.sh MNIST 8084 200 20 1 4 [4,5] 4
#./save_training_ova.sh MNIST 8085 200 20 1 4 [4,5] 5
#./save_training_ova.sh MNIST 8086 200 20 1 4 [6,7] 6
#./save_training_ova.sh MNIST 8087 200 20 1 4 [6,7] 7
#./save_training_ova.sh MNIST 8088 200 20 1 4 [8,9] 8
#./save_training_ova.sh MNIST 8089 200 20 1 4 [8,9] 9

# MNIST Non-IID 5 classes
#./save_training_ova.sh MNIST 8081 200 20 1 4 [0,1,2,3,4] 0
#./save_training_ova.sh MNIST 8082 200 20 1 4 [0,1,2,3,4] 1
#./save_training_ova.sh MNIST 8083 200 20 1 4 [0,1,2,3,4] 3
#./save_training_ova.sh MNIST 8084 200 20 1 4 [0,1,2,3,4] 2
#./save_training_ova.sh MNIST 8085 200 20 1 4 [0,1,2,3,4] 4
#./save_training_ova.sh MNIST 8086 200 20 1 4 [5,6,7,8,9] 5
#./save_training_ova.sh MNIST 8087 200 20 1 4 [5,6,7,8,9] 6
#./save_training_ova.sh MNIST 8088 200 20 1 4 [5,6,7,8,9] 7
#./save_training_ova.sh MNIST 8089 200 20 1 4 [5,6,7,8,9] 8
#./save_training_ova.sh MNIST 8080 200 20 1 4 [5,6,7,8,9] 9



# FMNIST IID

#                      <data>  <serverPort> <epochs> <numClients> <ova> <scenario>           <labels>       <modelType>
#./save_training_ova.sh FMNIST      8080        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       0
#./save_training_ova.sh FMNIST      8081        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       1
#./save_training_ova.sh FMNIST      8082        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       3
#./save_training_ova.sh FMNIST      8083        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       2
#./save_training_ova.sh FMNIST      8084        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       4
#./save_training_ova.sh FMNIST      8085        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       5
#./save_training_ova.sh FMNIST      8086        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       6
#./save_training_ova.sh FMNIST      8087        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       7
#./save_training_ova.sh FMNIST      8088        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       8
#./save_training_ova.sh FMNIST      8089        200        20        1       4        [0,1,2,3,4,5,6,7,8,9]       9

# FMNIST Non-IID 2 classes
#./save_training_ova.sh FMNIST 8080 200 20 1 4 [0,1] 0
#./save_training_ova.sh FMNIST 8081 200 20 1 4 [0,1] 1
#./save_training_ova.sh FMNIST 8082 200 20 1 4 [2,3] 3
#./save_training_ova.sh FMNIST 8083 200 20 1 4 [2,3] 2
#./save_training_ova.sh FMNIST 8084 200 20 1 4 [4,5] 4
#./save_training_ova.sh FMNIST 8085 200 20 1 4 [4,5] 5
#./save_training_ova.sh FMNIST 8086 200 20 1 4 [6,7] 6
#./save_training_ova.sh FMNIST 8087 200 20 1 4 [6,7] 7
#./save_training_ova.sh FMNIST 8088 200 20 1 4 [8,9] 8
#./save_training_ova.sh FMNIST 8089 200 20 1 4 [8,9] 9

# FMNIST Non-IID 5 classes
#./save_training_ova.sh FMNIST 8081 200 20 1 4 [0,1,2,3,4] 0
#./save_training_ova.sh FMNIST 8082 200 20 1 4 [0,1,2,3,4] 1
#./save_training_ova.sh FMNIST 8083 200 20 1 4 [0,1,2,3,4] 3
#./save_training_ova.sh FMNIST 8084 200 20 1 4 [0,1,2,3,4] 2
#./save_training_ova.sh FMNIST 8085 200 20 1 4 [0,1,2,3,4] 4
#./save_training_ova.sh FMNIST 8086 200 20 1 4 [5,6,7,8,9] 5
#./save_training_ova.sh FMNIST 8087 200 20 1 4 [5,6,7,8,9] 6
#./save_training_ova.sh FMNIST 8088 200 20 1 4 [5,6,7,8,9] 7
#./save_training_ova.sh FMNIST 8089 200 20 1 4 [5,6,7,8,9] 8
#./save_training_ova.sh FMNIST 8080 200 20 1 4 [5,6,7,8,9] 9
