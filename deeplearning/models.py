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
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


max_words = 10000
# time series i believe. set max_len to 1 for non-RNN
max_len = 1

class TransientModel:
    def __init__(self, tenant):
        plt.style.use('ggplot')

    def load_model(tenant):
        return keras.models.load_model('data/model/' + tenant)

    def load_labels(tenant):
        return json.load(open("data/label_map/tenant.json","r"))

    def load_vectorizer(tenant):
        vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=100)
        vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=max_len)
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

    def train(self, tenant, label_map, train_df, test_df):
        f = open("data/label_map/tenant.json","w+")
        f.write(json.dumps(label_map))
        f.close()
        train_df.drop(['_id'],axis=1,inplace=True)
        test_df.drop(['_id'],axis=1,inplace=True)
        train_df['label'].replace(label_map, inplace=True)
        test_df['label'].replace(label_map, inplace=True)
        self.neural_network(train_df, test_df)
        self.model.save('data\\model\\' + tenant)
        
    def predict(self, sentence):
        feature_vector = self.vectorize_sentence([sentence])
        print(feature_vector)
        prediction = self.model.predict(feature_vector)
        print(prediction)
        return prediction

    def initialize_vectorizer(self):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=max_len)
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

    def MLP(self):
        inputs = keras.layers.Input(name='inputs',shape=[max_len])
        layer = keras.layers.Embedding(max_words,50,input_length=max_len)(inputs)
        layer = keras.layers.Dense(256,name='FC1')(layer)
        layer = keras.layers.Activation('relu')(layer)
        layer = keras.layers.Dense(256,name='FC2')(layer)
        layer = keras.layers.Activation('relu')(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(18,name='out_layer')(layer)
        layer = keras.layers.Activation('sigmoid')(layer)
        model = keras.Model(inputs=inputs,outputs=layer)
        return model


    def RNN(self):
        inputs = keras.layers.Input(name='inputs',shape=[max_len])
        layer = keras.layers.Embedding(max_words,50,input_length=max_len)(inputs)
        layer = keras.layers.LSTM(64)(layer)
        layer = keras.layers.Dense(256,name='FC1')(layer)
        layer = keras.layers.Activation('relu')(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(18,name='out_layer')(layer)
        layer = keras.layers.Activation('sigmoid')(layer)
        model = keras.Model(inputs=inputs,outputs=layer)
        return model
        
    def neural_network(self, train_df, test_df):
        le = LabelEncoder()
        X_train = train_df['text']
        # y_train = train_df['label'].to_numpy().reshape(-1,1)
        y_train = le.fit_transform(train_df['label']).reshape(-1,1)
        X_test = test_df['text']
        # y_test = test_df['label'].to_numpy().reshape(-1,1)
        y_test = le.fit_transform(test_df['label']).reshape(-1,1)

        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(X_train)
        sequences = tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

        self.model = self.RNN()
        self.model.summary()
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
        history = self.model.fit(sequences_matrix,y_train,batch_size=128,epochs=10, validation_split=0.2)

        # inputs = keras.Input(shape=(100,), name='digits')
        # x = keras.layers.Dense(64, activation='relu', name='dense_1')(inputs)
        # x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        # outputs = keras.layers.Dense(4, activation='softmax', name='predictions')(x)
        # self.model = keras.Model(inputs=inputs, outputs=outputs)
        # self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        #       loss='sparse_categorical_crossentropy',
        #       metrics = ['accuracy'])
        # self.model.summary()
        # history = self.model.fit(train_dataset, epochs=30, validation_data=val_dataset)
        
        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

        train_sequences = tok.texts_to_sequences(X_train)
        train_sequences_matrix = sequence.pad_sequences(train_sequences,maxlen=max_len)

        loss, accuracy = self.model.evaluate(train_sequences_matrix, y_train)
        print("Training Accuracy:  {:.4f}".format(accuracy))
        loss, accuracy = self.model.evaluate(test_sequences_matrix, y_test)
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
