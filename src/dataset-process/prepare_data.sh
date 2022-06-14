#!/bin/bash


# configure the require datasets
DATASETS=('MNIST' 'CIFAR-10' 'FMNIST')

for DATASET_NAME in "${DATASETS[@]}"
do

	# prepare the directories
	./build_directory.sh $DATASET_NAME
	
	# download the dataset
	python3.9 get_data.py $DATASET_NAME

	# move the data to the respective directory
	./move_dataset.sh ../../$DATASET_NAME/Non-IID-distribution

done
