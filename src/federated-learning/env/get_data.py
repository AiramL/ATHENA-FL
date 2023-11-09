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
    
    

    total_files = len(listdir(results_path))
        
        
    for i,name in enumerate(listdir(results_path)):
        file_names["filename"+str(i+1)] = results_path+name
            
    for i in range(1,total_files+1):
        result_files["result"+str(i)] = open(file_names['filename'+str(i)], 'r')


    for i in range(1,total_files+1):
        file_lines["Lines"+str(i)] = result_files['result'+str(i)].readlines()
    
        
    for i in range(1,total_files+1):
        result_files['result'+str(i)].close()

    accuracies = []
    ac = []
    

    for i in range(total_files):
        accuracies.append([])
        ac.append([])

    if figureType == 1:
        for i in range(1,total_files+1):
            for line in file_lines['Lines'+str(i)]:
                if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                    accuracies[i-1].append(float(line.split(':')[1][1:].split(' ')[0]))        
    else:
        for i in range(1,total_files+1):
            for line in file_lines['Lines'+str(i)]:
                if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                    accuracies[i-1].append(float(line.split(':')[2].split(' ')[1]))
       
    for i in range(total_files):
        ac[i] = accuracies[i][:]

    ac_1 = [ele for ele in ac if ele != []]
    

    x1Mean = mean(ac_1,axis=0);
    x1Interval = std(ac_1,axis=0)*1.96/sqrt(10);
    x = arange(len(x1Mean))
        
    
    superior = x1Mean + x1Interval
    inferior = x1Mean - x1Interval
    
    superior_interval = []
    inferior_interval = []
    
    # limiting the confidence interval
    if figureType != 1:
        for item in superior:
            if item > 1:
                superior_interval.append(1)
            else:
                superior_interval.append(item)

    for item in inferior:
        if item < 0:
            inferior_interval.append(0)
        else:
            inferior_interval.append(item)
     
       
    return(x, x1Mean, superior_interval, inferior_interval)    
   
if __name__ == "__main__":

    #write_path = "../federated-learning/env/Only-5-Classes/SBRC_2023_accuracy_MNIST/"
    #read_path = "../federated-learning/env/Only-5-Classes/SBRC_2023_MNIST/"
    
    write_path = 'result_new_model_CIFAR/'
    read_path = 'result_new_model_CIFAR/'

#    for index in range(10):
#        print('before',index)
#        axis,mean,superior,inferior = file_to_list(figureType=0,epochs=200,results_path=read_path+'model-'+str(index)+'/')
#
#        print(index)
#        with open(write_path+'mean_model'+str(index),'w') as writer:
#            csv_data = ''
#            for data in mean:
#                csv_data += str(data)+','
#            writer.writelines(csv_data[:-1])
#    
#        with open(write_path+'superior_model'+str(index),'w') as writer:
#            csv_data = ''
#            for data in superior:
#                csv_data += str(data)+','
#            writer.writelines(csv_data[:-1])
#            
#        with open(write_path+'inferior_model'+str(index),'w') as writer:
#            csv_data = ''
#            for data in inferior:
#                csv_data += str(data)+','
#            writer.writelines(csv_data[:-1])
#       
    if len(argv) > 1:
        index = int(argv[1])
    else:
        index = 2


    axis,mean,superior,inferior = file_to_list(figureType=0,epochs=200,results_path=read_path+'model-'+str(index)+'/')

    with open(write_path+'mean_model'+str(index),'w') as writer:
        csv_data = ''
        for data in mean:
            csv_data += str(data)+','
        writer.writelines(csv_data[:-1])
    
    with open(write_path+'superior_model'+str(index),'w') as writer:
        csv_data = ''
        for data in superior:
            csv_data += str(data)+','
        writer.writelines(csv_data[:-1])
        
    with open(write_path+'inferior_model'+str(index),'w') as writer:
        csv_data = ''
        for data in inferior:
            csv_data += str(data)+','
        writer.writelines(csv_data[:-1])



