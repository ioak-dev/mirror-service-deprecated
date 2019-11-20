import tensorflow as tf
from tensorflow import keras

class Model:
    network_name = None
    def __init__(self, network_name):
        self.network_name = network_name

    def train(self, train_df, test_df):
        print(self.network_name)
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        model.fit(train_df, epochs=10)