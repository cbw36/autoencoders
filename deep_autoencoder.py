from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from simple_autoencoder import Autoencoder

ENCODING_DIM = 32
INPUT_DIM = (784,)
(x_train, _), (x_test, _) = mnist.load_data()


class DeepAutoencoder(Autoencoder):
    def __init__(self, x_train, x_test, encoding_dim, input_dim=INPUT_DIM, sparse=False, encoder=None, decoder=None):
        super().__init__(x_train, x_test, encoding_dim, input_dim=input_dim, sparse=sparse, encoder=encoder, decoder=decoder)

    def createAutoencoder(self):
        self.encoded_layer = layers.Dense(128, activation='relu')(self.input)
        self.encoded_layer = layers.Dense(64, activation='relu')(self.encoded_layer)
        self.encoded_layer = layers.Dense(32, activation='relu')(self.encoded_layer)

        decoded_layer = layers.Dense(64, activation='relu')(self.encoded_layer)
        decoded_layer = layers.Dense(128, activation='relu')(decoded_layer)
        decoded_layer = layers.Dense(784, activation='sigmoid')(decoded_layer)

        self.autoencoder = keras.Model(self.input, decoded_layer)

dae = DeepAutoencoder(x_train, x_test, ENCODING_DIM, sparse=True)
dae.prepareData()
dae.createInput()
dae.createAutoencoder()
dae.createEncoder()
dae.runModel()
dae.visualizeResults()