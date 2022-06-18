#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#

import numpy as np

from pickle import load
from sklearn.utils import shuffle
from ova_processing import binary_labels

def load_data_federated(dataset_name,clientID,numClients,basicNN,modelType,trPer):

    if clientID <= numClients//2:     
        # first class
        x_train1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class0Train','rb')),
                dtype=np.float32)
        y_train1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class0TrainLabel','rb')),
                dtype=np.float32)
        x_test1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class0Test','rb')),
                dtype=np.float32)
        y_test1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class0TestLabel','rb')),
                dtype=np.float32)
        
        x_train1 = x_train1[int(len(x_train1)/numClients*(clientID-1)):int(len(x_train1)/numClients*clientID)]
        y_train1 = y_train1[int(len(y_train1)/numClients*(clientID-1)):int(len(y_train1)/numClients*clientID)]
        x_test1 = x_test1[int(len(x_test1)/numClients*(clientID-1)):int(len(x_test1)/numClients*clientID)]
        y_test1 = y_test1[int(len(y_test1)/numClients*(clientID-1)):int(len(y_test1)/numClients*clientID)]
        
        
        # second class
        x_train2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class1Train','rb')),
                dtype=np.float32)
        y_train2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class1TrainLabel','rb')),
                dtype=np.float32)
        x_test2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class1Test','rb')),
                dtype=np.float32)
        y_test2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class1TestLabel','rb')),
                dtype=np.float32)
        
    
        x_train2 = x_train2[int(len(x_train2)/numClients*(clientID-1)):int(len(x_train2)/numClients*clientID)]
        y_train2 = y_train2[int(len(y_train2)/numClients*(clientID-1)):int(len(y_train2)/numClients*clientID)]
        x_test2 = x_test2[int(len(x_test2)/numClients*(clientID-1)):int(len(x_test2)/numClients*clientID)]
        y_test2 = y_test2[int(len(y_test2)/numClients*(clientID-1)):int(len(y_test2)/numClients*clientID)]
    
    
    
        # third class
        x_train3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class2Train','rb')),
                dtype=np.float32)
        y_train3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class2TrainLabel','rb')),
                dtype=np.float32)
        x_test3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class2Test','rb')),
                dtype=np.float32)
        y_test3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class2TestLabel','rb')),
                dtype=np.float32)
        
        x_train3 = x_train3[int(len(x_train3)/numClients*(clientID-1)):int(len(x_train3)/numClients*clientID)]
        y_train3 = y_train3[int(len(y_train3)/numClients*(clientID-1)):int(len(y_train3)/numClients*clientID)]
        x_test3 = x_test3[int(len(x_test3)/numClients*(clientID-1)):int(len(x_test3)/numClients*clientID)]
        y_test3 = y_test3[int(len(y_test3)/numClients*(clientID-1)):int(len(y_test3)/numClients*clientID)]
       
        
        # fourth class
        x_train4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class3Train','rb')),
                dtype=np.float32)
        y_train4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class3TrainLabel','rb')),
                dtype=np.float32)
        x_test4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class3Test','rb')),
                dtype=np.float32)
        y_test4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class3TestLabel','rb')),
                dtype=np.float32)
        
        x_train4 = x_train4[int(len(x_train4)/numClients*(clientID-1)):int(len(x_train4)/numClients*clientID)]
        y_train4 = y_train4[int(len(y_train4)/numClients*(clientID-1)):int(len(y_train4)/numClients*clientID)]
        x_test4 = x_test4[int(len(x_test4)/numClients*(clientID-1)):int(len(x_test4)/numClients*clientID)]
        y_test4 = y_test4[int(len(y_test4)/numClients*(clientID-1)):int(len(y_test4)/numClients*clientID)]
    
        
        # fifth class
        x_train5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class4Train','rb')),
                dtype=np.float32)
        y_train5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class4TrainLabel','rb')),
                dtype=np.float32)
        x_test5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class4Test','rb')),
                dtype=np.float32)
        y_test5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class4TestLabel','rb')),
                dtype=np.float32)
        
    
        x_train5 = x_train5[int(len(x_train5)/numClients*(clientID-1)):int(len(x_train5)/numClients*clientID)]
        y_train5 = y_train5[int(len(y_train5)/numClients*(clientID-1)):int(len(y_train5)/numClients*clientID)]
        x_test5 = x_test5[int(len(x_test5)/numClients*(clientID-1)):int(len(x_test5)/numClients*clientID)]
        y_test5 = y_test5[int(len(y_test5)/numClients*(clientID-1)):int(len(y_test5)/numClients*clientID)]

    
    
    else:
        # first class
        x_train1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class5Train','rb')),
                dtype=np.float32)
        y_train1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class5TrainLabel','rb')),
                dtype=np.float32)
        x_test1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class5Test','rb')),
                dtype=np.float32)
        y_test1 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class5TestLabel','rb')),
                dtype=np.float32)
        
        
        x_train1 = x_train1[int(len(x_train1)/numClients*(clientID-1)):int(len(x_train1)/numClients*clientID)]
        y_train1 = y_train1[int(len(y_train1)/numClients*(clientID-1)):int(len(y_train1)/numClients*clientID)]
        x_test1 = x_test1[int(len(x_test1)/numClients*(clientID-1)):int(len(x_test1)/numClients*clientID)]
        y_test1 = y_test1[int(len(y_test1)/numClients*(clientID-1)):int(len(y_test1)/numClients*clientID)]
        
        # second class
        x_train2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class6Train','rb')),
                dtype=np.float32)
        y_train2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class6TrainLabel','rb')),
                dtype=np.float32)
        x_test2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class6Test','rb')),
                dtype=np.float32)
        y_test2 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class6TestLabel','rb')),
                dtype=np.float32)
        
    
        x_train2 = x_train2[int(len(x_train2)/numClients*(clientID-1)):int(len(x_train2)/numClients*clientID)]
        y_train2 = y_train2[int(len(y_train2)/numClients*(clientID-1)):int(len(y_train2)/numClients*clientID)]
        x_test2 = x_test2[int(len(x_test2)/numClients*(clientID-1)):int(len(x_test2)/numClients*clientID)]
        y_test2 = y_test2[int(len(y_test2)/numClients*(clientID-1)):int(len(y_test2)/numClients*clientID)]
    
        # third class
        x_train3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class7Train','rb')),
                dtype=np.float32)
        y_train3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class7TrainLabel','rb')),
                dtype=np.float32)
        x_test3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class7Test','rb')),
                dtype=np.float32)
        y_test3 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class7TestLabel','rb')),
                dtype=np.float32)
        
       
        x_train3 = x_train3[int(len(x_train3)/numClients*(clientID-1)):int(len(x_train3)/numClients*clientID)]
        y_train3 = y_train3[int(len(y_train3)/numClients*(clientID-1)):int(len(y_train3)/numClients*clientID)]
        x_test3 = x_test3[int(len(x_test3)/numClients*(clientID-1)):int(len(x_test3)/numClients*clientID)]
        y_test3 = y_test3[int(len(y_test3)/numClients*(clientID-1)):int(len(y_test3)/numClients*clientID)]
        
        # fourth class
        x_train4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class8Train','rb')),
                dtype=np.float32)
        y_train4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class8TrainLabel','rb')),
                dtype=np.float32)
        x_test4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class8Test','rb')),
                dtype=np.float32)
        y_test4 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class8TestLabel','rb')),
                dtype=np.float32)
        
    
        x_train4 = x_train4[int(len(x_train4)/numClients*(clientID-1)):int(len(x_train4)/numClients*clientID)]
        y_train4 = y_train4[int(len(y_train4)/numClients*(clientID-1)):int(len(y_train4)/numClients*clientID)]
        x_test4 = x_test4[int(len(x_test4)/numClients*(clientID-1)):int(len(x_test4)/numClients*clientID)]
        y_test4 = y_test4[int(len(y_test4)/numClients*(clientID-1)):int(len(y_test4)/numClients*clientID)]
        
        # fifth class 
        x_train5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class9Train','rb')),
                dtype=np.float32)
        y_train5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/train/class9TrainLabel','rb')),
                dtype=np.float32)
        x_test5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class9Test','rb')),
                dtype=np.float32)
        y_test5 = np.asarray(load(open('../../../../datasets/'+dataset_name+'/Non-IID-distribution/test/class9TestLabel','rb')),
                dtype=np.float32)
        
        x_train5 = x_train5[int(len(x_train5)/numClients*(clientID-1)):int(len(x_train5)/numClients*clientID)]
        y_train5 = y_train5[int(len(y_train5)/numClients*(clientID-1)):int(len(y_train5)/numClients*clientID)]
        x_test5 = x_test5[int(len(x_test5)/numClients*(clientID-1)):int(len(x_test5)/numClients*clientID)]
        y_test5 = y_test5[int(len(y_test5)/numClients*(clientID-1)):int(len(y_test5)/numClients*clientID)]
    

    x_train = np.concatenate((x_train1,x_train2,x_train3,x_train4,x_train5,x_test1,x_test2,x_test3,x_test4,x_test5))/255 
    y_train = np.concatenate((y_train1,y_train2,y_train3,y_train4,y_train5,y_test1,y_test2,y_test3,y_test4,y_test5))


    # If it is a basic NN we train a One-versus-All models
    if basicNN:
        y_train = binary_labels(y_train,modelType)
    
    x_train, y_train = shuffle(x_train, y_train, random_state=47527)
    
    trSize = int(len(x_train)*trPer)
    
    return x_train[:trSize], y_train[:trSize], x_train[trSize:], y_train[trSize:]
