#!/bin/bash


find . -name "*result*" -type f -size -4k -delete

for i in $(seq 0 9)
do 
	mkdir model-$i
	mv *simple-model-$i-epochs-200-clients-40-client* model-$i	
done
