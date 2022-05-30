import flwr as fl
import tensorflow as tf
from pickle import load
import numpy as np

# client configuration
serverPort = '8080'
modelType = 1
clientID = 1

if len(argv) >= 2:
    modelType = int(argv[1])

if len(argv) >= 3:
    serverPort = argv[2]

if len(argv) >= 4:
    clientID = int(argv[3])

# Loading the dataset

# first class
x_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0Train','rb')),dtype=np.float32)
y_train1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class0TrainLabel','rb')),dtype=np.float32)
x_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0Test','rb')),dtype=np.float32)
y_test1 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class0TestLabel','rb')),dtype=np.float32)

x_train1 = x_train1[1000*(clientID-1):1000*clientID]
y_train1 = y_train1[1000*(clientID-1):1000*clientID]
x_test1 = x_test1[100*(clientID-1):100*clientID]
y_test1 = y_test1[100*(clientID-1):100*clientID]


# second class
x_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1Train','rb')),dtype=np.float32)
y_train2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class1TrainLabel','rb')),dtype=np.float32)
x_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1Test','rb')),dtype=np.float32)
y_test2 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class1TestLabel','rb')),dtype=np.float32)

x_train2 = x_train2[1000*(clientID-1):1000*clientID]
y_train2 = y_train2[1000*(clientID-1):1000*clientID]
x_test2 = x_test2[100*(clientID-1):100*clientID]
y_test2 = y_test2[100*(clientID-1):100*clientID]


# third class

x_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2Train','rb')),dtype=np.float32)
y_train3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class2TrainLabel','rb')),dtype=np.float32)
x_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2Test','rb')),dtype=np.float32)
y_test3 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class2TestLabel','rb')),dtype=np.float32)


x_train3 = x_train3[1000*(clientID-1):1000*clientID]
y_train3 = y_train3[1000*(clientID-1):1000*clientID]
x_test3 = x_test3[100*(clientID-1):100*clientID]
y_test3 = y_test3[100*(clientID-1):100*clientID]

# fourth class

x_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3Train','rb')),dtype=np.float32)
y_train4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class3TrainLabel','rb')),dtype=np.float32)
x_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3Test','rb')),dtype=np.float32)
y_test4 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class3TestLabel','rb')),dtype=np.float32)

x_train4 = x_train4[1000*(clientID-1):1000*clientID]
y_train4 = y_train4[1000*(clientID-1):1000*clientID]
x_test4 = x_test4[100*(clientID-1):100*clientID]
y_test4 = y_test4[100*(clientID-1):100*clientID]

# fifth class

x_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4Train','rb')),dtype=np.float32)
y_train5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/train/class4TrainLabel','rb')),dtype=np.float32)
x_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4Test','rb')),dtype=np.float32)
y_test5 = np.asarray(load(open('../../../../datasets/CIFAR-10/Non-IID-distribution/test/class4TestLabel','rb')),dtype=np.float32)


x_train5 = x_train5[1000*(clientID-1):1000*clientID]
y_train5 = y_train5[1000*(clientID-1):1000*clientID]
x_test5 = x_test5[100*(clientID-1):100*clientID]
y_test5 = y_test5[100*(clientID-1):100*clientID]

# create the training data
x_train = np.concatenate((x_train1,x_train2,x_train3,x_train4,x_train5))


# create the test data
x_test = np.concatenate((x_test1,x_test2,x_test3,x_test4,x_test5))

if modelType == 0:
    y_train = np.concatenate((np.ones(len(y_train1)),np.zeros(len(y_train2)*4)))
    y_test = np.concatenate((np.ones(len(y_test1)),np.zeros(len(y_test2)*4)))

elif modelType == 1:
    y_train = np.concatenate((np.ones(len(y_train2)),np.zeros(len(y_train1)*4)))
    y_test = np.concatenate((np.ones(len(y_test2)),np.zeros(len(y_test1)*4)))

elif modelType == 2:
    y_train = np.concatenate((np.ones(len(y_train3)),np.zeros(len(y_train1)*4)))
    y_test = np.concatenate((np.ones(len(y_test3)),np.zeros(len(y_test1)*4)))

elif modelType == 3:
    y_train = np.concatenate((np.ones(len(y_train4)),np.zeros(len(y_train1)*4)))
    y_test = np.concatenate((np.ones(len(y_test4)),np.zeros(len(y_test1)*4)))

elif modelType == 4:
    y_train = np.concatenate((np.ones(len(y_train5)),np.zeros(len(y_train1)*4)))
    y_test = np.concatenate((np.ones(len(y_test5)),np.zeros(len(y_test1)*4)))


model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(32, 32, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

class CifarClient(fl.client.NumPyClient):
	
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5,steps_per_epoch=512)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
        return loss, len(x_test), {"accuracy": accuracy}

	

fl.client.start_numpy_client("[::]:"+serverPort, client=CifarClient())


