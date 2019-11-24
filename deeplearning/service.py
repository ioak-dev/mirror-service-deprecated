import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json, time
from deeplearning.models import Model, ModelContainer
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

def get_dataset(tenant, csv_data):
    dataset = get_collection(tenant, 'dataset_test').find({})
    df = pd.DataFrame(list(dataset))
    if df.size == 0:
        return (204, {})
    else:
        df.pop('_id')
        return (200, {
            'dimension': df.shape,
            'data': json.loads(df.to_json(orient='records'))
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

def create_model(tenant, network_name):
    model = Model(network_name)
    ModelContainer.add(tenant, network_name, model)
    return (200, {})

def remove_model(tenant, network_name):
    ModelContainer.remove(tenant, network_name)
    return (200, {})
    
def featuretext_to_vector(tenant, network_name):
    if CELERY_BROKER_URL is None:
        tasks.vectorize(tenant, network_name)
        return (200, {})
    else:
        task_result = tasks.vectorize.delay(tenant, network_name)
        return (200, {'async_task_id': task_result.id})
        # res = AsyncResult(task_result.id)
        # response = res.collect()
        # for a, v in response:
        #     print(a)
        #     print(v)

def train_model(tenant, network_name):
    if CELERY_BROKER_URL is None:
        tasks.train_model(tenant, network_name)
        return (200, {})
    else:
        task_result = tasks.train_model.delay(tenant, network_name)
        return (200, {'async_task_id': task_result.id})

def predict(tenant, network_name, sentence):
    model = ModelContainer.get(tenant, network_name)
    sentence = nlp_utils.clean_text(sentence)
    prediction = model.predict(sentence)
    print(prediction)
    ranks = prediction[0].argsort().argsort()
    categories = get_collection(tenant, 'category').find({})
    label_map = list_to_dict(list(categories), 'value', 'name')
    print(type(prediction[0]))
    print(type(prediction[0][0]))
    outcome = []
    for i in range(len(prediction[0])):
        outcome.append({
            'label': label_map.get(i),
            'rank': str(ranks[i]),
            'probability': str(prediction[0][i])
        })
    return (200, {'sentence': sentence, 'prediction': outcome})
    # return (200, {'sentence': sentence})
