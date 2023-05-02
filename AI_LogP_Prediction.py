#Importamos todas las librerias necesarias y comprobamos si podremos usar la GPU
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from pandas import *
from PIL import Image as im
import matplotlib.image
print("GPU available: ", tf.test.is_gpu_available())
print(tf.__version__)

#Se lee los datos desde los archivos
DataTest = read_csv("Smiles_Test.csv")
DataTrain = read_csv("Smiles_Train.csv")
SmilesTest = DataTest['SMILES_Test'].tolist()
LogPStringTest = DataTest['LogP_Test'].tolist()
SmilesTrain = DataTrain['SMILES'].tolist()
LogPStringTrain = DataTrain['logP'].tolist()
LogPTrain = np.array(LogPStringTrain)
LogPTest = np.array(LogPStringTest)

#Guardamos los datos en un nuevo arreglo rectangular de 1x784
SmilesTrainLen = []
SmilesTestLen = []
i = 0
j = 0
for x in SmilesTrain:
    SmilesTrainLen.append(len(SmilesTrain[i]))
    i += 1
for y in SmilesTest:
    SmilesTestLen.append(len(SmilesTest[j]))
    j += 1

a = 0
MoleculaAsciiTest = []
for x in range(len(SmilesTest)):
    b = 0
    s = SmilesTest[x]
    MolLen = len(SmilesTest[x])
    Arreglo1 = []
    for y in range(784):
        if (b < MolLen):
            Arreglo1.append(ord(s[b]))
        else:
            Arreglo1.append(0)
        b += 1
    MoleculaAsciiTest.append(Arreglo1)
    a += 1

a = 0
MoleculaAsciiTrain = []
for x in range(len(SmilesTrain)):
    b = 0
    s = SmilesTrain[x]
    MolLen = len(SmilesTrain[x])
    Arreglo1 = []
    for y in range(784):
        if (b < MolLen):
            Arreglo1.append(ord(s[b]))
        else:
            Arreglo1.append(0)
        b += 1
    MoleculaAsciiTrain.append(Arreglo1)
    a += 1

#Modulo que convierte la matriz en una de 28x28
for y in range(len(SmilesTrain)):
    SmilesTrainImg = []
    SmilesTrainImg = MoleculaAsciiTrain[y]
    SmilesTrainImg = np.array(SmilesTrainImg)
    SmilesTrainImg = np.reshape(SmilesTrainImg, (28,28))
    i = str(y)
    #matplotlib.image.imsave('SmilesTrainImages/Train'+i+'.png',SmilesTrainImg)

for y in range(len(SmilesTest)):
    SmilesTestImg = []
    SmilesTestImg = MoleculaAsciiTest[y]
    SmilesTestImg = np.array(SmilesTestImg)
    SmilesTestImg = np.reshape(SmilesTestImg, (28,28))
    i = str(y)
    #matplotlib.image.imsave('SmilesTestImages/Test'+i+'.png',SmilesTestImg)

#Convertimos en arreglo numerico y en un tensor
MoleculaAsciiTrain = np.asarray(MoleculaAsciiTrain)
MoleculaAsciiTest = np.asarray(MoleculaAsciiTest)
LogPTrain = np.asarray(LogPTrain)
LogPTest = np.asarray(LogPTest)
MoleculeTensorTrain = tf.convert_to_tensor(MoleculaAsciiTrain)
MoleculeTensorTest = tf.convert_to_tensor(MoleculaAsciiTest)

#Modelo para tener entrada de 28x28 de 255 canales
modelo = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=[784]),
    tf.keras.layers.Conv2D(32, (3,3),input_shape=(28,28,255), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3),input_shape=(28,28,255), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
    tf.keras.layers.Dense(units=1, activation=tf.nn.relu)
])

modelo.compile(
    optimizer='adam',
    #loss='mean_squared_error',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("Iniciando Entrenamiento...")
modelo.fit(
    SmilesTrain, LogPTrain, epochs = 1,
    verbose = False
)
print("Entrenamiento finailzado exitosamente")

test_lost, test_accuracy = modelo.evaluate(MoleculaAsciiTest)
print("Resultado en las pruebas: ", test_accuracy)

print("Hagamos una prediccion")
resultado = modelo.predict([MoleculaAsciiTrain])
print("El resultado es " + str(resultado))