import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pickle import load
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


from dataset_operations import *


X = read_dataset('../../../datasets/CICDataset/processed_data/pre_processed_dataframes/features_1')
Y = read_dataset('../../../datasets/CICDataset/processed_data/pre_processed_dataframes/label_1_binary')

Y = Y.replace([-1],0)

x_tr,x_test,y_tr,y_test = train_test_split(X,Y,test_size=0.2)

x_train,x_val,y_train,y_val = train_test_split(x_tr,y_tr,test_size=0.3)

x_train.drop("SimillarHTTP", axis='columns', inplace=True)
x_val.drop("SimillarHTTP", axis='columns', inplace=True)
x_test.drop("SimillarHTTP", axis='columns', inplace=True)

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_val = np.asarray(y_val).astype('float32').reshape((-1,1))


#classes = len(Y.unique())
classes = 1

# Model from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8681044

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1024,input_shape=(len(x_train.columns),), activation='relu', kernel_initializer='normal',activity_regularizer=tf.keras.regularizers.l1(0.01)))
model.add(tf.keras.layers.BatchNormalization(input_shape=(model.output_shape),synchronized=True))
model.add(tf.keras.layers.Dropout(0.1,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dense(768, activation='relu', kernel_initializer='normal',activity_regularizer=tf.keras.regularizers.l1(0.01))),
model.add(tf.keras.layers.BatchNormalization(synchronized=True,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dropout(0.1,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='normal',input_shape=(model.output_shape)))
model.add(tf.keras.layers.BatchNormalization(synchronized=True,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dropout(0.1,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='normal',input_shape=(model.output_shape),activity_regularizer=tf.keras.regularizers.l1(0.01)))
model.add(tf.keras.layers.BatchNormalization(synchronized=True,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dropout(0.1,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_initializer='normal',input_shape=(model.output_shape),activity_regularizer=tf.keras.regularizers.l1(0.01)))
model.add(tf.keras.layers.BatchNormalization(synchronized=True,input_shape=(model.output_shape)))
model.add(tf.keras.layers.Dropout(0.1,input_shape=(model.output_shape)))
#model.add(tf.keras.layers.Dense(classes, activation='softmax'))
model.add(tf.keras.layers.Dense(classes, activation='sigmoid'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1,weight_decay=0.01)

#model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=optimizer,loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])


bs = 1024

#print(x_train[1])

#model.fit(x_train, y_train,validation_data=(x_val,y_val),epochs=10,batch_size=bs,steps_per_epoch=int(len(x_train)//bs))
model.fit(x_train, y_train,validation_data=(x_val,y_val),epochs=10,batch_size=bs)


loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
