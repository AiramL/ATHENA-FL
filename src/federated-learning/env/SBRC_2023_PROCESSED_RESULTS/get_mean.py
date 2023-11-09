import numpy as np

from math import sqrt

data = []

for i in range(10):
    data.append(float(open("mean_model"+str(i),"r").readlines()[0].split(',')[-1]))

print(np.mean(data))
print(np.std(data)*1.96/sqrt(40))


    

