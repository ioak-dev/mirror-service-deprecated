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
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, network_name):
        self.network_name = network_name
        plt.style.use('ggplot')

    def train(self, df, label_list):
        self.print_dataset_proportion([df])
        label_dict = collection_utils.list_to_dict(label_list, 'name', 'value')
        df['label'] = df['label'].map(label_dict)
        df['text'] = df['text'].apply(nlp_utils.clean_text)
        train_df, remain_df = train_test_split(df, train_size=0.7, stratify=df['label'])
        val_df, test_df = train_test_split(remain_df, train_size=0.1, stratify=remain_df['label'])
        self.print_dataset_proportion([df, train_df, val_df, test_df])
        
        self.initialize_vectorizer(train_df['text'].values)
        X_train = self.vectorize(train_df)
        X_val = self.vectorize(val_df)
        X_test = self.vectorize(test_df)
        y_train = train_df['label'].values
        y_val = val_df['label'].values
        y_test = test_df['label'].values

        # self.logistic_regression((X_train, X_val, X_test), (y_train, y_val, y_test))
        self.neural_network((X_train, X_val, X_test), (y_train, y_val, y_test), df['label'].nunique())

    def predict(self, sentence):
        feature_vector = self.vectorize_sentence([sentence])
        print(feature_vector)
        prediction = self.model.predict(feature_vector)
        print(prediction)
        return prediction

    def initialize_vectorizer(self, sentences):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=100)
        self.vectorizer.fit(sentences)
    
    def vectorize(self, df):
        sentences = df['text'].values
        return self.vectorizer.transform(sentences).toarray()

    def vectorize_sentence(self, sentence):
        return self.vectorizer.transform(sentence).toarray()

    def print_dataset_proportion(self, df_list):
        for df in df_list:
            print(df.groupby('label').count())

    def logistic_regression(self, X, y):
        classifier = LogisticRegression()
        classifier.fit(X[0], y[0])
        score = classifier.score(X[2], y[2])
        print('Accuracy = ', score)

    def neural_network(self, X, y, label_count):
        y_train = y[0].astype('float32')
        y_val = y[1].astype('float32')
        y_test = y[2].astype('float32')
        X_train = X[0].astype('float32')
        X_val = X[1].astype('float32')
        X_test = X[2].astype('float32')
        print(X_train.shape, y_train.shape, type(X_train), type(y_train))

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(64)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(64)

        inputs = keras.Input(shape=(100,), name='digits')
        x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
        x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        outputs = keras.layers.Dense(label_count, activation='softmax', name='predictions')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
        self.model.summary()
        history = self.model.fit(train_dataset, epochs=30, validation_data=val_dataset)
        
        loss, accuracy = self.model.evaluate(train_dataset)
        print("Training Accuracy:  {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate(val_dataset)
        print("Validation Accuracy:  {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate(test_dataset)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
        # self.plot_history(history)
    
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
