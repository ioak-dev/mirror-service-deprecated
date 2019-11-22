import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json
from deeplearning.models import Model, ModelContainer
import library.nlp_utils as nlp_utils
from sklearn.model_selection import train_test_split

DATABASE_URI = os.environ.get('DATABASE_URI')

def add_dataset(tenant, csv_data):
    df = pd.read_csv(StringIO(csv_data))
    print(df.groupby('label').count())
    df = df.dropna()
    df, remain_df = train_test_split(df, train_size=10000, stratify=df['label'])
    print(df.groupby('label').count())
    get_collection(tenant, 'dataset').insert(df.to_dict('records'))
    # os.makedirs('dataset/' + tenant)
    # df.to_csv('dataset/' + tenant + '/dataset.csv', mode='a+', index=False, header=None)
    return (200, {'added_dimension': df.shape})

def get_dataset(tenant, csv_data):
    dataset = get_collection(tenant, 'dataset').find({})
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
    response = get_collection(tenant, 'dataset').remove({})
    return (200, {'count': response['n']})

def create_model(tenant, network_name):
    model = Model(network_name)
    ModelContainer.add(tenant, network_name, model)
    return (200, {})

def remove_model(tenant, network_name):
    ModelContainer.remove(tenant, network_name)
    return (200, {})
    
def train_model(tenant, network_name):
    create_model(tenant, network_name)
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
    

def predict(tenant, network_name, sentence):
    model = ModelContainer.get(tenant, network_name)
    sentence = nlp_utils.clean_text(sentence)
    prediction = model.predict(sentence)
    print(type(prediction))
    print(type(list(prediction)))
    return (200, {'sentence': sentence, 'prediction': str(list(prediction))})
    # return (200, {'sentence': sentence})