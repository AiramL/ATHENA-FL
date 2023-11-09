import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, std, arange
from math import sqrt
from os import listdir
from sys import argv

# figureType = 1 -> print loss
# figureType = otherwise -> print accuracy

# language = 1 -> print in portuguese-br
# language = otherwise -> print in english

def file_to_list(figureType=0,epochs=100,results_path='../federated-learning/env/Only-5-Classes/results/'):

    
    file_names = {}
    result_files = {}
    file_lines = {}
    
    

     
        
    with open(results_path, 'r') as result_data:
        file_lines = result_data.readlines()


    accuracies = []

    if figureType == 1:
        for line in file_lines:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies.append(float(line.split(':')[1][1:].split(' ')[0]))        
    else:
        for line in file_lines:
            if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                accuracies.append(float(line.split(':')[2].split(' ')[1]))
       

    return accuracies 


if __name__ == "__main__":

    #write_path = "../federated-learning/env/Only-5-Classes/SBRC_2023_accuracy_MNIST/"
    #read_path = "../federated-learning/env/Only-5-Classes/SBRC_2023_MNIST/"
    
    write_path = 'results_SBRC/'
    read_path = 'results/result-CIFAR-10-complex-model-epochs-200-clients-20-client'

    if len(argv) > 1:
        index = int(argv[1])
    else:
        index = 2

    for i in range(1,21):
        acc = file_to_list(figureType=0,epochs=200,results_path=read_path+str(i))
    
        with open(write_path+'acc_client'+str(i),'w') as writer:
            csv_data = ''
            for data in acc:
                csv_data += str(data)+','
            writer.writelines(csv_data[:-1])
        
    


