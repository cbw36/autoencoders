from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard

from simple_autoencoder import Autoencoder

ENCODING_DIM = 32
INPUT_DIM = (28, 28, 1)
(x_train, _), (x_test, _) = mnist.load_data()


class ConvolutionalAutoencoder(Autoencoder):
    def __init__(self, x_train, x_test, encoding_dim, input_dim=INPUT_DIM, sparse=False, encoder=None, decoder=None):
        super().__init__(x_train, x_test, encoding_dim, input_dim=input_dim, sparse=sparse, encoder=encoder, decoder=decoder)

    def createAutoencoder(self):
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(self.input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        self.encoded_layer = layers.MaxPooling2D((2, 2), padding='same')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(self.encoded_layer)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded_layer = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = keras.Model(self.input, decoded_layer)

    def prepareData(self):
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.x_train = np.reshape(self.x_train, ((len(self.x_train),) + self.input_dim))
        self.x_test = np.reshape(self.x_test, ((len(self.x_test),) + self.input_dim))




cae = ConvolutionalAutoencoder(x_train, x_test, ENCODING_DIM, sparse=True)
cae.prepareData()
cae.createInput()
cae.createAutoencoder()
cae.createEncoder()
cae.runModel(epochs=50, batch_size=128)
cae.visualizeResults()