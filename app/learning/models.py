# import tensorflow as tf
# from tensorflow import keras
# import tensorflow_hub as hub
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
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn import metrics
import pickle
from pathlib import Path
import os

from library.db_connection_factory import get_collection
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing import sequence


max_words = 20000
# time series i believe. set max_len to 1 for non-RNN
max_len = 1
vectorizer_type = 'count'
ml_model = 'lr'
n = 1

trained_models_dir = Path('artifacts')
# trained_models_dir ='./app/learning/tenant'

all_models = {}
all_vectorizers = {}
all_labels = {}

class TransientModel:
    def __init__(self, tenant):
        plt.style.use('ggplot')

    def load_model(self, tenant):
        print(trained_models_dir)
        print(tenant)
        # return keras.models.load_model('data/model/' + tenant)
        all_models[tenant] = pickle.load(open(trained_models_dir / tenant / 'model.sav', 'rb'))
        all_vectorizers[tenant] = pickle.load(open(trained_models_dir / tenant / 'vectorizer.sav', 'rb'))
        all_labels[tenant] = pickle.load(open(trained_models_dir / tenant / 'label.sav', 'rb'))
        return "trained model read successful"

    def load_labels(self, tenant):
        return json.load(open("data/label_map/tenant.json","r"))

    def load_vectorizer(self, tenant):
        vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=max_len)
        f = open('learning/vocab.txt', 'r', encoding='utf-8')
        vectorizer.fit(f.readlines())
        return vectorizer

    def train(self, tenant):
        #df = dataset_service.get_dataset(tenant)
        df = pd.DataFrame(list(get_collection(tenant, 'dataset_train').find({})))
        computed_labels = {}
        index = 0
        for label_item in df['label']:
            try:
                computed_labels[label_item.lstrip().rstrip()]
            except KeyError:
                computed_labels[label_item.lstrip().rstrip()] = index
                index = index + 1
        df.replace(computed_labels, inplace=True)

        df.text=df.text.fillna(' ')
        X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, train_size=0.8, stratify=df.label)

        vectorizer=TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2', ngram_range=(1, 2), stop_words='english', use_idf=True, max_df=.1, max_features=max_words)
        vectorizer = vectorizer.fit(X_train)
        X_train=vectorizer.transform(X_train)
        X_test=vectorizer.transform(X_test)
        classifier = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial', max_iter=200)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_proba = classifier.predict_proba(X_test)

        count_misclassified = (y_test != y_pred).sum()
        print('Misclassified samples: {}'.format(count_misclassified))
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print('Accuracy: {:.2f}'.format(accuracy))

        print(ml_model, vectorizer_type, n)
        matches = 0
        for x in range(0, y_pred_proba.shape[0]):
            if y_test.iloc[x] in np.argsort(y_pred_proba[x])[::-1][:n]:
                matches = matches + 1
        top_n_accuracy = matches / y_test.shape[0]
        print('Misclassified samples: {}'.format(y_test.shape[0] - matches))
        print('Accuracy: {:.2f}'.format(top_n_accuracy))
        
        pathname=os.path.join(trained_models_dir,tenant)
        Path(pathname).mkdir(parents=True, exist_ok=True)
        pickle.dump(classifier, open(trained_models_dir / tenant / 'model.sav', 'wb'))
        pickle.dump(vectorizer, open(trained_models_dir / tenant / 'vectorizer.sav', 'wb'))
        pickle.dump(dict((v, k) for k, v in computed_labels.items()), open(trained_models_dir / tenant / 'label.sav', 'wb'))
        return "training successful"
        
    def prediction(self, tenant, data):
        pred_in = all_vectorizers[tenant].transform([data])
        outcome = all_models[tenant].predict_proba(pred_in)
        print(outcome[0])
        return (outcome[0], all_labels[tenant])

    def initialize_vectorizer(self):
        self.vectorizer = CountVectorizer(min_df=0, lowercase=False, max_features=max_len)
        f = open('./app/learning/vocab.txt', 'r', encoding='utf-8')
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
