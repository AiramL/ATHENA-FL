import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean, std, arange
from math import sqrt

# figureType = 1 -> print loss
# figureType = otherwise -> print accuracy

# language = 1 -> print in portuguese
# language = otherwise -> print in english

def plot_image(figureType=0,language=1,numFiles=10,dataset_name="CIFAR-10"):

    
    file_names = {}
    result_files = {}
    file_lines = {}

    for i in range(1,numFiles+1):
        file_names["filename"+str(i)] = '../federated-learning/env/Only-5-Classes/results/result-'+dataset_name+'-complex-model-client'+str(i)


    for i in range(1,numFiles+1):
        result_files["result"+str(i)] = open(file_names['filename'+str(i)], 'r')


    for i in range(1,numFiles+1):
        file_lines["Lines"+str(i)] = result_files['result'+str(i)].readlines()

    
    for i in range(1,numFiles+1):
        result_files['result'+str(i)].close()
    

    accuracies = []
    ac = []

    for i in range(numFiles):
        accuracies.append([])
        ac.append([])

    if figureType == 1:
        for i in range(1,numFiles+1):
            for line in file_lines['Lines'+str(i)]:
                if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                    accuracies[i-1].append(float(line.split(':')[1][1:].split(' ')[0]))        
    else:
        for i in range(1,numFiles+1):
            for line in file_lines['Lines'+str(i)]:
                if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
                    accuracies[i-1].append(float(line.split(':')[2].split(' ')[1]))
       
    for i in range(numFiles):
        ac[i] = accuracies[i][:75]

#    x1Mean = mean(accuracies,axis=0);
#    x1Interval = std(accuracies,axis=0)*1.96/sqrt(10);
    x1Mean = mean(ac,axis=0);
    x1Interval = std(ac,axis=0)*1.96/sqrt(10);
    x = arange(len(x1Mean))


    # 2 class experiment cluster
    #filename1 = '../federated-learning-env/create-models/results/result-simple-model-2-client1'
    #filename2 = '../federated-learning-env/create-models/results/result-simple-model-2-client2'
    #filename3 = '../federated-learning-env/create-models/results/result-simple-model-2-client3'
    #filename4 = '../federated-learning-env/create-models/results/result-simple-model-2-client4'
    #filename5 = '../federated-learning-env/create-models/results/result-simple-model-2-client5'
    #filename1 = '../federated-learning-env/create-models/Only-5-Classes/results/result-simple-model-2-client1'
    #filename2 = '../federated-learning-env/create-models/Only-5-Classes/results/result-simple-model-2-client2'
    #filename3 = '../federated-learning-env/create-models/Only-5-Classes/results/result-simple-model-2-client3'
    #filename4 = '../federated-learning-env/create-models/Only-5-Classes/results/result-simple-model-2-client4'
    #filename5 = '../federated-learning-env/create-models/Only-5-Classes/results/result-simple-model-2-client5'

   # results1 = open(filename1, 'r')
   # results2 = open(filename2, 'r')
   # results3 = open(filename3, 'r')
   # results4 = open(filename4, 'r')
   # results5 = open(filename5, 'r')

   # Lines1 = results1.readlines()
   # Lines2 = results2.readlines()
   # Lines3 = results3.readlines()
   # Lines4 = results4.readlines()
   # Lines5 = results5.readlines()

   # accuracies2 = [[],[],[],[],[]]

   # if figureType == 1:
   #     for line in Lines1:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[0].append(float(line.split(':')[1][1:].split(' ')[0]))        
   #     for line in Lines2:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[1].append(float(line.split(':')[1][1:].split(' ')[0]))
   #     for line in Lines3:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[2]
   #     for line in Lines4:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[3].append(float(line.split(':')[1][1:].split(' ')[0]))
   #     for line in Lines5:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[4].append(float(line.split(':')[1][1:].split(' ')[0]))

   # else:
   #     for line in Lines1:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[0].append(float(line.split(':')[2].split(' ')[1]))
   #     for line in Lines2:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[1].append(float(line.split(':')[2].split(' ')[1]))
   #     for line in Lines3:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[2].append(float(line.split(':')[2].split(' ')[1]))
   #     for line in Lines4:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[3].append(float(line.split(':')[2].split(' ')[1]))
   #     for line in Lines5:
   #         if (not 'Epoch' in line) and (not 'ETA' in line) and (line != '\n') and (not '=' in line):
   #             accuracies2[4].append(float(line.split(':')[2].split(' ')[1]))

   # x2Mean = mean(accuracies2[:13],axis=0);
   # x2Interval = std(accuracies2[:13],axis=0)*1.96/sqrt(10);



    if language == 1 and figureType == 1:
        plt.plot(x, x1Mean, 'r-', label='Aprendizado Federado Tradicional')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)    
   #     plt.plot(x, x2Mean, 'b-',label='Proposta Atual')
   #     plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Perda no Teste', fontsize=16)
        plt.xlabel('Época', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()

    elif language == 1 and figureType != 1:
        plt.plot(x, x1Mean, 'r-', label='Aprendizado Federado Tradicional')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        #plt.plot(x, x2Mean, 'b-',label='Proposta Atual')
        #plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Acurácia', fontsize=16)
        plt.xlabel('Época', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()

        
    elif language != 1 and figureType == 1:
        plt.plot(x, x1Mean, 'r-', label='Traditional Federated Learning')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        #plt.plot(x, x2Mean, 'b-',label='Our Proposal')
        #plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Test Loss', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()


    elif language != 1 and figureType != 1:
        plt.plot(x, x1Mean, 'r-', label='Traditional Federated Learning')
        plt.fill_between(x, x1Mean - x1Interval, x1Mean + x1Interval, color='r', alpha=0.2)
        #plt.plot(x, x2Mean, 'b-',label='Our Proposal')
        #plt.fill_between(x, x2Mean - x2Interval, x2Mean + x2Interval, color='b', alpha=0.2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.xlabel('Epoch', fontsize=16)
        plt.legend()
        #plt.savefig('teste1.png')
        plt.show()

plot_image(0,1)



