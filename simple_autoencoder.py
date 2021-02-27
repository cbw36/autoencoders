from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

ENCODING_DIM = 32
INPUT_DIM = (784,)
(x_train, _), (x_test, _) = mnist.load_data()



class Autoencoder:
    def __init__(self, x_train, x_test, encoding_dim=ENCODING_DIM, input_dim=INPUT_DIM, sparse=False, encoder=None, decoder=None):
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.x_train = x_train
        self.x_test = x_test
        self.sparse = sparse

    def createInput(self):
        self.input = keras.Input(shape=self.input_dim)

    def createAutoencoder(self):
        if self.sparse:
            self.encoded_layer = layers.Dense(self.encoding_dim, activation='relu',
                                activity_regularizer=regularizers.l1(10e-5))(self.input)
        else:
            self.encoded_layer = layers.Dense(self.encoding_dim, activation='relu')(self.input)

        decoded_layer = layers.Dense(self.input_dim[0], activation="sigmoid")(self.encoded_layer)

        self.autoencoder = keras.Model(self.input, decoded_layer)

    def createEncoder(self):
        self.encoder = keras.Model(self.input, self.encoded_layer)

    def prepareData(self):
        self.x_train = self.x_train.astype('float32') / 255.
        self.x_test = self.x_test.astype('float32') / 255.
        self.x_train = np.reshape(self.x_train, ((len(self.x_train),) + self.input_dim))
        self.x_test = np.reshape(self.x_test, ((len(self.x_test),) + self.input_dim))

    def runModel(self, epochs=100, batch_size=256):
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        self.autoencoder.fit(self.x_train, self.x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(self.x_test, self.x_test))

    def visualizeResults(self):
        # encoded_imgs = self.encoder.predict(self.x_test)
        decoded_imgs = self.autoencoder.predict(self.x_test)

        n = 10  # How many digits we will display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
#
# ae = Autoencoder(x_train, x_test, sparse=True)
# ae.prepareData()
# ae.createInput()
# ae.createAutoencoder()
# ae.createEncoder()
# ae.runModel()
# ae.visualizeResults()