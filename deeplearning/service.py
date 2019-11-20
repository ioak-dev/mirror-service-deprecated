import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json

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
    response = get_collection(tenant, 'dataset_' + datatype).remove({})
    return (200, {'count': response['n']})