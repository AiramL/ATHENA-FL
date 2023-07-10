#!/bin/bash



for i in $(seq 0 9)
do 
	mkdir model-$i
	mv *simple-model-$i-epochs* model-$i	
done
