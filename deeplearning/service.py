import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json
from deeplearning.models import Model, ModelContainer

DATABASE_URI = os.environ.get('DATABASE_URI')

def add_dataset(tenant, csv_data, datatype):
    df = pd.read_csv(StringIO(csv_data))
    get_collection(tenant, 'dataset_' + datatype).insert(df.to_dict('records'))
    # os.makedirs('dataset/' + tenant)
    # df.to_csv('dataset/' + tenant + '/' + datatype + '.csv', mode='a+', index=False, header=None)
    return (200, {'added_dimension': df.shape})

def get_dataset(tenant, csv_data, datatype):
    dataset = get_collection(tenant, 'dataset_' + datatype).find({})
    df = pd.DataFrame(list(dataset))
    if df.size == 0:
        return (204, {})
    else:
        df.pop('_id')
        return (200, {
            'dimension': df.shape,
            'data': json.loads(df.to_json(orient='records'))
            })


def clear_dataset(tenant, csv_data, datatype):
    response = get_collection(tenant, 'dataset_'  + datatype).remove({})
    return (200, {'count': response['n']})

def create_model(tenant, network_name):
    model = Model(network_name)
    ModelContainer.add(tenant, network_name, model)
    return (200, {})

def remove_model(tenant, network_name):
    ModelContainer.remove(tenant, network_name)
    return (200, {})
    
def train_model(tenant, network_name):
    model = ModelContainer.get(tenant, network_name)
    if model == None:
        return (204, {'error': 'model not present uner network name [' + network_name + ']'})
    else:
        train_dataset = get_collection(tenant, 'dataset_train').find({})
        test_dataset = get_collection(tenant, 'dataset_test').find({})
        train_df = pd.DataFrame(list(train_dataset))
        test_df = pd.DataFrame(list(test_dataset))
        if train_df.size == 0 or test_df.size == 0:
            return (204, {'error': 'train / test dataset is not present'})
        else:
            train_df.pop('_id')
            test_df.pop('_id')
            model.train(train_df, test_df)
            return (200, {'train_dimension': train_df.shape, 'test_dimension': test_df.shape})
    