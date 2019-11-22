import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json
from deeplearning.models import Model, ModelContainer

DATABASE_URI = os.environ.get('DATABASE_URI')

def add_category(tenant, data):
    id = get_collection(tenant, 'category').insert_many(data)
    return (200, {'_id': str(id)})

def get_category_all(tenant):
    categories = get_collection(tenant, 'category').find({})
    categories_list = list(categories)
    for category in categories_list:
        del category['_id']
    return (200, {'data': categories_list})
