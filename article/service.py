import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json

DATABASE_URI = os.environ.get('DATABASE_URI')

def add_category(tenant, data):
    categoryId = get_collection(tenant, 'category').find().sort({'name':-1})
    #print(list(categoryId))
    id = get_collection(tenant, 'category').insert_many(data)
    return (200, {'_id': str(id)})

def get_category_all(tenant):
    categories = get_collection(tenant, 'category').find({})
    categories_list = list(categories)
    for category in categories_list:
        del category['_id']
        del category['categoryId']
    return (200, {'data': categories_list})
