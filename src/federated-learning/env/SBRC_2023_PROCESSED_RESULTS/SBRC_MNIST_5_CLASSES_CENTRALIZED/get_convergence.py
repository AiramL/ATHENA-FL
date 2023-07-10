
tol = 0.001;
#model_epoch = [];
#model_acc = [];


def read_csv(model_name):
    with open(model_name,"r") as csv:
        return csv.readlines()[0].split(',')


for model in [0,9]:#range(10):
#model = 2;
    avg = read_csv("mean_model"+str(model))
    #avg = list(avg);
    valor_anterior = float(avg[0]);
    for i in range(1,201):
        if (valor_anterior*(1-tol) < float(avg[i]) and  valor_anterior*(1+tol) > float(avg[i])):
#            model_epoch = [model_epoch i];
            print(i)
            print(model)
            #%valor_anterior;
            #model_acc = [model_acc avg(i)];
            print(avg[i])
            break
        else:
            valor_anterior = float(avg[i]);
       # end
    #end
#%end 

#mean(model_epoch)
#std(model_epoch)
