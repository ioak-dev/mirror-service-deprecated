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

class TransientModel:
    def __init__(self, tenant):
        plt.style.use('ggplot')

    def load_model(tenant):
        return keras.models.load_model('data/model/' + tenant)

    def load_vectorizer(tenant):
        vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=100)
        f = open('deeplearning/vocab.txt', 'r', encoding='utf-8')
        vectorizer.fit(f.readlines())
        return vectorizer

    def tensor_to_tuple(self, line):
        features = tf.io.parse_single_example(
        line,
        features={
            'features': tf.io.FixedLenFeature([100], tf.float32),
            'label': tf.io.FixedLenFeature([1], tf.int64)
        })
        return features['features'], features['label']

    def train(self, tenant):
        train_dataset = tf.data.TFRecordDataset(filenames = ['data/tfrecords/' + tenant + '_' + 'train' + '.tfrecords']).map(self.tensor_to_tuple).shuffle(1000).batch(100)
        val_dataset = tf.data.TFRecordDataset(filenames = ['data/tfrecords/' + tenant + '_' + 'val' + '.tfrecords']).map(self.tensor_to_tuple).shuffle(1000).batch(100)
        test_dataset = tf.data.TFRecordDataset(filenames = ['data/tfrecords/' + tenant + '_' + 'test' + '.tfrecords']).map(self.tensor_to_tuple).shuffle(1000).batch(100)
        self.neural_network(train_dataset, val_dataset, test_dataset)
        self.model.save('data/model/' + tenant)

    def predict(self, sentence):
        feature_vector = self.vectorize_sentence([sentence])
        print(feature_vector)
        prediction = self.model.predict(feature_vector)
        print(prediction)
        return prediction

    def initialize_vectorizer(self):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=100)
        f = open('deeplearning/vocab.txt', 'r', encoding='utf-8')
        self.vectorizer.fit(f.readlines())
        # self.vectorizer.fit(df['text'].values)
    
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

    def neural_network(self, train_dataset, val_dataset, test_dataset):
        inputs = keras.Input(shape=(100,), name='digits')
        x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
        x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        outputs = keras.layers.Dense(self.label_count, activation='softmax', name='predictions')(x)
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
    def add(tenant, model, vectorizer):
        ModelContainer.instances[tenant] = {'model': model, 'vectorizer': vectorizer}

    @staticmethod
    def get(tenant):
        if tenant in ModelContainer.instances:
            return ModelContainer.instances[tenant].get('model'), ModelContainer.instances[tenant].get('vectorizer')
        else:
            return

    @staticmethod
    def remove(tenant):
        if tenant in ModelContainer.instances:
            del ModelContainer.instances[tenant]
