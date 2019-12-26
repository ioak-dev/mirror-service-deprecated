import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json, time
from deeplearning.models import TransientModel, ModelContainer
import library.nlp_utils as nlp_utils
from sklearn.model_selection import train_test_split
import library.collection_utils as collection_utils
import tensorflow as tf
import deeplearning.tasks as tasks
from celery.result import AsyncResult

DATABASE_URI = os.environ.get('DATABASE_URI')
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL')
CELERY_BROKER_URL = None
def add_dataset(tenant, csv_data):
    df = pd.read_csv(StringIO(csv_data))
    df = df.dropna()
    train_df, test_df = train_test_split(df, train_size=0.8, stratify=df['label'])
    get_collection(tenant, 'dataset_train').insert(train_df.to_dict('records'))
    get_collection(tenant, 'dataset_test').insert(test_df.to_dict('records'))
    return (200, {'dimension': {
                    'train': train_df.shape,
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
    test_response = get_collection(tenant, 'dataset_test').remove({})
    return (200, {'count': {
                        'train': train_response,
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

def load_labels(tenant):
    label_map = TransientModel.load_labels(tenant)
    return (200, label_map)

def predict(tenant, sentence):
    model, vectorizer = ModelContainer.get(tenant)
    sentence = nlp_utils.clean_text(sentence)
    feature_vector = vectorizer.transform([sentence]).toarray()
    prediction = model.predict(feature_vector)
    print(prediction)
    ranks = prediction[0].argsort().argsort()
    label_map = collection_utils.dict_invert(TransientModel.load_labels(tenant))
    outcome = []
    for i in range(len(prediction[0])):
        outcome.append({
            'label': label_map.get(i),
            'rank': str(ranks[i]),
            'probability': str(prediction[0][i])
        })
    return (200, {'sentence': sentence, 'prediction': outcome})
    # return (200, {'sentence': sentence})
