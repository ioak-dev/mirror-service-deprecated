import tensorflow as tf
from tensorflow import keras

class Model:
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
        # model.fit(train_df, epochs=10)

class ModelContainer:
    instances = {}
    
    @staticmethod
    def add(tenant, network_name, model):
        ModelContainer.instances[tenant] = {network_name: model}

    @staticmethod
    def get(tenant, network_name):
        if tenant in ModelContainer.instances and network_name in ModelContainer.instances[tenant]:
            print('PRESENT****')
            return ModelContainer.instances[tenant][network_name]
        else:
            print('ABSENT****')

    @staticmethod
    def remove(tenant, network_name):
        if tenant in ModelContainer.instances and network_name in ModelContainer.instances[tenant]:
            del ModelContainer.instances[tenant][network_name]
