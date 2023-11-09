from numpy import mean, std 

tol = 0.001;
#model_epoch = [];
#model_acc = [];


def read_csv(model_name):
    with open(model_name,"r") as csv:
        return csv.readlines()[0].split(',')

epoc =[]

for model in range(1,21):
#model = 2;
    avg = read_csv("results_SBRC/acc_client"+str(model))
    #avg = list(avg);
    valor_anterior = float(avg[0]);
    for i in range(1,201):
        if (valor_anterior*(1-tol) < float(avg[i]) and  valor_anterior*(1+tol) > float(avg[i])):
#            model_epoch = [model_epoch i];
#            print(i)
            epoc.append(i)
#            print(model)\
            #%valor_anterior;
            #model_acc = [model_acc avg(i)];
#            print(avg[i])
            break
        else:
            valor_anterior = float(avg[i]);
       # end
    #end
#%end 

print(mean(epoc))
print(std(epoc))
