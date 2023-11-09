#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#

import tensorflow as tf
from tensorflow.keras import layers,models


def build_model(basicNN, dataset_name,nn_id=1):
    if dataset_name == "CIFAR-10":
        # Verify if we are training a robust model or OvA models
        if not basicNN:
            if nn_id == 1:
                model = tf.keras.applications.MobileNet((32, 32, 3), classes=10, weights=None,dropout=0.01)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
            elif nn_id == 2:
                model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

            elif nn_id == 3:
                model = tf.keras.applications.Xception(input_shape=(None,32, 32, 3), classes=10, weights=None,include_top=False)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
        
        
        # If it is a basic NN we train a One-versus-All models
        else:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(32, 32, 3),padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
            
    elif dataset_name == "MNIST":
        # Verify if we are training a robust model or OvA models
        if not basicNN:
            if nn_id == 1:
                model = tf.keras.applications.MobileNet((32, 32, 1), classes=10, weights=None,dropout=0.01)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                       loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
            elif nn_id == 2:
                model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 1), include_top=False,classes=10,alpha=0.7, weights=None)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

            elif nn_id == 3:
                model = tf.keras.applications.Xception((32, 32, 1), classes=10, weights=None,include_top=False)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
        
        
        # If it is a basic NN we train a One-versus-All models
        else:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(32, 32,1),padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
           
    elif dataset_name == "FMNIST":
            # Verify if we are training a robust model or OvA models
        if not basicNN:
            if nn_id == 1:
                model = tf.keras.applications.MobileNet((32, 32, 1), classes=10, weights=None,dropout=0.005)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                        loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            elif nn_id == 2:
                model = tf.keras.applications.MobileNetV2((32, 32, 1), classes=10, weights=None)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])

            elif nn_id == 3:
                model = tf.keras.applications.Xception(include_top=False,input_shape=(32, 32, 1), classes=10, weights=None)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])
        
        
        # If it is a basic NN we train a One-versus-All models
        else:
            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1),padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu',padding='same'))
            model.add(layers.Flatten())
            model.add(layers.Dense(64, activation='relu'))
            model.add(layers.Dense(2))
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    
    return model
