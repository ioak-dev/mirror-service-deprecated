import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json, time
from deeplearning.models import TransientModel, ModelContainer
import library.nlp_utils as nlp_utils
from sklearn.model_selection import train_test_split
from library.collection_utils import list_to_dict
import tensorflow as tf
import deeplearning.tasks as tasks
from celery.result import AsyncResult

DATABASE_URI = os.environ.get('DATABASE_URI')
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')

def add_dataset(tenant, csv_data):
    df = pd.read_csv(StringIO(csv_data))
    df = df.dropna()
    print('before')
    train_df, remain_df = train_test_split(df, train_size=0.7, stratify=df['label'])
    print('after one')
    print(remain_df.groupby('label').count())
    val_df, test_df = train_test_split(remain_df, train_size=0.1, stratify=remain_df['label'])
    print('after two')
    get_collection(tenant, 'dataset_train').insert(train_df.to_dict('records'))
    get_collection(tenant, 'dataset_val').insert(val_df.to_dict('records'))
    get_collection(tenant, 'dataset_test').insert(test_df.to_dict('records'))
    return (200, {'dimension': {
                    'train': train_df.shape,
                    'val': val_df.shape,
                    'test': test_df.shape
                    }
                }
            )

def get_dataset(tenant):
    train_count = get_collection(tenant, 'dataset_train').find({}).count()
    val_count = get_collection(tenant, 'dataset_val').find({}).count()
    test_count = get_collection(tenant, 'dataset_test').find({}).count()
    return (200, {
        'train': train_count,
        'val': val_count,
        'test': test_count
        })


def clear_dataset(tenant, csv_data):
    train_response = get_collection(tenant, 'dataset_train').remove({})
    val_response = get_collection(tenant, 'dataset_val').remove({})
    test_response = get_collection(tenant, 'dataset_test').remove({})
    return (200, {'count': {
                        'train': train_response,
                        'val': val_response,
                        'test': test_response
                    }
                }
            )

def remove_model(tenant):
    ModelContainer.remove(tenant)
    return (200, {})
    
def train_model(tenant):
    if CELERY_BROKER_URL is None:
        tasks.train_model(tenant)
        return (200, {})
    else:
        task_result = tasks.train_model.delay(tenant)
        return (200, {'async_task_id': task_result.id})

# def train_model(tenant):
#     if CELERY_BROKER_URL is None:
#         tasks.train_model(tenant)
#         return (200, {})
#     else:
#         task_result = tasks.train_model.delay(tenant)
#         return (200, {'async_task_id': task_result.id})

def load_model(tenant):
    model = TransientModel.load_model(tenant)
    vectorizer = TransientModel.load_vectorizer(tenant)
    ModelContainer.add(tenant, model, vectorizer)
    return (200, {})

def predict(tenant, sentence):
    model, vectorizer = ModelContainer.get(tenant)
    sentence = nlp_utils.clean_text(sentence)
    feature_vector = vectorizer.transform([sentence]).toarray()
    prediction = model.predict(feature_vector)
    print(prediction)
    ranks = prediction[0].argsort().argsort()
    categories = get_collection(tenant, 'category').find({})
    label_map = list_to_dict(list(categories), 'value', 'name')
    outcome = []
    for i in range(len(prediction[0])):
        outcome.append({
            'label': label_map.get(i),
            'rank': str(ranks[i]),
            'probability': str(prediction[0][i])
        })
    return (200, {'sentence': sentence, 'prediction': outcome})
    # return (200, {'sentence': sentence})
