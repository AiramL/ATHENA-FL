# Authors: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformática e Automação (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#
# usage: python split_class.py <number-of-clients>

from sys import argv

from split_class_mnist.py import get_MNIST
from split_class_fashion.py import get_FMNIST
from split_class_cifar10.py import get_CIFAR10


# check the dataset to download
if len(argv) > 1:
    dataset_name = argv[1]
else:
    dataset_name = "MNIST"


if dataset_name == "MNIST":
    get_MNIST()

elif dataset_name == "FMNIST":
    get_FMNIST()

elif dataset_name == "CIFAR10":
    get_CIFAR10()

else:
    print("We do not implement yet the processing of this dataset.")

