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


class DenoisingAutoencoder(Autoencoder):
    def __init__(self, x_train, x_test, encoding_dim, input_dim=INPUT_DIM, sparse=False, encoder=None, decoder=None):
        super().__init__(x_train, x_test, encoding_dim, input_dim=input_dim, sparse=sparse, encoder=encoder, decoder=decoder)

    def createAutoencoder(self):
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.input)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        self.encoded_layer = layers.MaxPooling2D((2, 2), padding='same')(x)

        # At this point the representation is (7, 7, 32)

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(self.encoded_layer)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded_layer = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = keras.Model(self.input, decoded_layer)

    def prepareData(self):
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.x_train = np.reshape(self.x_train, ((len(self.x_train),) + self.input_dim))
        self.x_test = np.reshape(self.x_test, ((len(self.x_test),) + self.input_dim))

        noise_factor = 0.5
        self.x_train_noisy = self.x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=self.x_train.shape)
        self.x_test_noisy = self.x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=self.x_test.shape)

        self.x_train_noisy = np.clip(self.x_train_noisy, 0., 1.)
        self.x_test_noisy = np.clip(self.x_test_noisy, 0., 1.)

    def visualizeNoisyImages(self):
        n = 10
        plt.figure(figsize=(20, 2))
        for i in range(1, n + 1):
            ax = plt.subplot(1, n, i)
            plt.imshow(self.x_test_noisy[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def runModel(self, epochs=100, batch_size=256):
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.fit(self.x_train_noisy, self.x_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             shuffle=True,
                             validation_data=(self.x_test_noisy, self.x_test))


dae = DenoisingAutoencoder(x_train, x_test, ENCODING_DIM, sparse=True)
dae.prepareData()
dae.visualizeNoisyImages()
dae.createInput()
dae.createAutoencoder()
dae.createEncoder()
dae.runModel(epochs=100, batch_size=128)
dae.visualizeResults()