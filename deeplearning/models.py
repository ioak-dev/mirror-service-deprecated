import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import library.collection_utils as collection_utils
import library.nlp_utils as nlp_utils
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import pandas as pd

class Model:
    def __init__(self, network_name):
        self.network_name = network_name
        plt.style.use('ggplot')

    def train(self, train_df, test_df, label_list):
        label_dict = collection_utils.list_to_dict(label_list, 'name', 'value')
        train_df['label'] = train_df['label'].map(label_dict)
        test_df['label'] = test_df['label'].map(label_dict)
        train_df['text'] = train_df['text'].apply(nlp_utils.clean_text)
        test_df['text'] = test_df['text'].apply(nlp_utils.clean_text)

        self.initialize_vectorizer(train_df['text'].values)
        X_train = self.vectorize(train_df['text'].values)
        y_train = train_df['label'].values
        X_test = self.vectorize(test_df['text'].values)
        y_test = test_df['label'].values

        # self.logistic_regression(X_train, y_train, X_test, y_test)
        self.neural_network(X_train, y_train, X_test, y_test)

    def initialize_vectorizer(self, sentences):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=100)
        self.vectorizer.fit(sentences)
    
    def vectorize(self, sentences):
        return self.vectorizer.transform(sentences).toarray()

    def logistic_regression(self, X_train, y_train, X_test, y_test):
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print('Accuracy = ', score)

    def neural_network(self, X_train, y_train, X_test, y_test):
        # training_df = pd.DataFrame.from_records(X_train)
        # testing_df = pd.DataFrame.from_records(X_test)
        # training_df_y = pd.DataFrame.from_records(y_train.reshape((-1,1)))
        # testing_df_y = pd.DataFrame.from_records(y_test.reshape((-1,1)))

        # train_dataset = (tf.data.Dataset.from_tensor_slices((
        #     tf.cast(training_df.values, tf.float32),
        #     tf.cast(training_df_y[0].values, tf.int32)
        # )))
        # test_dataset = (tf.data.Dataset.from_tensor_slices((
        #     tf.cast(testing_df.values, tf.float32),
        #     tf.cast(testing_df_y[0].values, tf.int32)
        # )))

        # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # print(x_train.shape, y_train.shape, type(x_train), type(y_train))
        # x_train = x_train.reshape(60000, 784).astype('float32') / 255
        # x_test = x_test.reshape(10000, 784).astype('float32') / 255

        # y_train = y_train.astype('float32')
        # y_test = y_test.astype('float32')
        # print(x_train.shape, y_train.shape, type(x_train), type(y_train))
        # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        # train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
        # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)

        # inputs = keras.Input(shape=(784,), name='digits')
        # x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
        # x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        # outputs = keras.layers.Dense(10, activation='softmax', name='predictions')(x)

        # model = keras.Model(inputs=inputs, outputs=outputs)
        # model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        #       loss='sparse_categorical_crossentropy')
        # model.summary()
        # model.fit(train_dataset, epochs=3)
        # return

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_dim=X_train.shape[1], activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        model.summary()
        history = model.fit(X_train, y_train,
            epochs=100,
            verbose=False,
            validation_data=(X_test, y_test))
        
        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        self.plot_history(history)
    
    def plot_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

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
