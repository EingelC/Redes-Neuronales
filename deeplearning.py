#Instalar matplotlib python -m pip install -U matplotlib
#Instalar pip python -m pip install -U pip
#Instalar pip install tensorflow datasets
#Instalar pip install matplotlib
#Instalar pip install --upgrade pip

import tensorflow as tf
from tensorflow import keras
import numpy as np

celsius = np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit = np.array([-40,14,32,46,59,72,100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)
print("Comenzando Entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Modelo Entrenado")

print("Hagamos una prediccion")
resultado = modelo.predict([38.0])
print("El resultado es " + str(resultado) + " fahrenheit!")