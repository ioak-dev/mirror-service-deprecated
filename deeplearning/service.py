import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json
from deeplearning.models import Model, ModelContainer
import library.nlp_utils as nlp_utils
from sklearn.model_selection import train_test_split
from library.collection_utils import list_to_dict
import tensorflow as tf

DATABASE_URI = os.environ.get('DATABASE_URI')

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
    create_model(tenant, network_name)
    model = ModelContainer.get(tenant, network_name)
    # model.initialize_vectorizer(pd.DataFrame(list(get_collection(tenant, 'dataset_train').find({}))))
    print('initializing vectorizer')
    model.initialize_vectorizer()
    print('initialized vectorizer')
    categories = get_collection(tenant, 'category').find({})
    label_map = list_to_dict(list(categories), 'name', 'value')
    write_to_tfrecord(tenant, 'val', label_map, model.vectorizer)
    label_count = write_to_tfrecord(tenant, 'train', label_map, model.vectorizer)
    write_to_tfrecord(tenant, 'test', label_map, model.vectorizer)

    model.label_count = label_count
    
    # dataset = tf.data.TFRecordDataset(filenames = ['data/' + tenant + '_' + 'val' + '.tfrecords'])
    # # dataset.batch(1)
    # i = 0
    # for item in dataset.take(5):
    #     parsed = tf.train.Example.FromString(item.numpy())
    #     i = i+1
    #     # print(parsed)
    #     # print(i)
    return (200, {})

def write_to_tfrecord(tenant, datatype, label_map, vectorizer):
    print('vectorizing', datatype)
    label_count = set()
    cursor = get_collection(tenant, 'dataset_' + datatype).find({})
    with tf.io.TFRecordWriter('data/' + tenant + '_' + datatype + '.tfrecords') as writer:
        i = 0
        for item in cursor:
            i = i + 1
            features = vectorizer.transform([item['text']]).toarray()[0]
            label = label_map.get(item['label'])
            # print(vector[0].shape)
            # print(label)
            label_count.add(label)
            example = tf.train.Example()
            example.features.feature["features"].float_list.value.extend(features)
            example.features.feature["label"].int64_list.value.append(label)
            writer.write(example.SerializeToString())
    print('label_count', datatype, len(label_count))
    print('record_count', datatype, i)
    return len(label_count)

def train_model_using_df(tenant, network_name):
    model = ModelContainer.get(tenant, network_name)
    if model == None:
        return (204, {'error': 'model not present uner network name [' + network_name + ']'})
    else:
        dataset = get_collection(tenant, 'dataset').find({})
        df = pd.DataFrame(list(dataset))
        if df.size == 0:
            return (204, {'error': 'train / test dataset is not present'})
        else:
            df.pop('_id')
            categories = get_collection(tenant, 'category').find({})
            model.train(df, list(categories))
            return (200, {'dimension': df.shape})


def train_model(tenant, network_name):
    model = ModelContainer.get(tenant, network_name)
    if model == None:
        return (204, {'error': 'model not present uner network name [' + network_name + ']'})
    else:
        model.train(tenant)
        return (200, {})

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
