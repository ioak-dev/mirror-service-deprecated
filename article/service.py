import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import pandas as pd
from io import StringIO
import os, json

DATABASE_URI = os.environ.get('DATABASE_URI')

def add_category(tenant, data):
    category_id = get_collection(tenant, 'category').find({})
    l_category = 0
    for cid in category_id:
        local_Category = cid['categoryId']
        l_category = int(local_Category)+1
    for value in data:
        value['categoryId'] = l_category
        l_category = l_category+1
    id = get_collection(tenant, 'category').insert_many(data)
    return (200, {'_id': str(id)})

def get_category_all(tenant):
    categories = get_collection(tenant, 'category').find({})
    categories_list = list(categories)
    for category in categories_list:
        del category['_id']
        del category['categoryId']
    return (200, {'data': categories_list})
