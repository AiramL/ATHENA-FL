# Authors: Gustavo Franco Camilo and Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python server.py <numbers-of-rounds>  <server-port>

import tensorflow as tf
import flwr as fl
from sys import argv

numRounds = 400
serverPort ='8080'
fr_fit=0.2
num_clients_fit = 2
num_clients = 2

if len(argv) >= 2:
    numRounds = int(argv[1])

if len(argv) >= 3:
    #serverPort = argv[2strategy = fl.server.strategy.FedAvg(min_available_clients=2,fraction_fit=0.5,fraction_evaluate=1.0)
    serverPort = argv[2]
if len(argv) >= 4:
    fr_fit=float(argv[3])

if len(argv) >= 5:
    num_client_fit = int(argv[4])

if len(argv) >= 6:
    num_clients = int(argv[5])

print(fr_fit)

strategy = fl.server.strategy.FedAvg(min_available_clients=num_clients,min_fit_clients=num_clients_fit,fraction_fit=fr_fit,fraction_eval=1.0)

#fl.server.start_server(config=fl.server.ServerConfig(num_rounds=numRounds),server_address='[::]:'+serverPort,strategy=strategy)
fl.server.start_server(config={'num_rounds':numRounds},server_address='[::]:'+serverPort,strategy=strategy)



