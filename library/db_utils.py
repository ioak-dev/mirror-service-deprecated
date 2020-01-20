from pymongo import MongoClient
from library.db_connection_factory import get_collection
from bson.objectid import ObjectId
import os
from datetime import datetime

DATABASE_URI = os.environ.get('DATABASE_URI')

def find(tenant, collection_name, search_criteria, user_id = None):
    data = get_collection(tenant, collection_name).find(search_criteria)
    data = list(data)
    data = clean_array(data)
    return data

def upsert(tenant, collection_name, data, user_id = None):
    now = datetime.now()
    data['lastModifiedBy'] = user_id
    data['lastModifiedAt'] = now
    if data.get('id') is None:
        data['createdBy'] = user_id
        data['createdAt'] = now
        response = get_collection(tenant, collection_name).insert_one(data)
        record = get_collection(tenant, collection_name).find_one({'_id': response.inserted_id})
        return clean_object(record)
    else:
        data['_id'] = ObjectId(data.get('id'))
        del data['id']
        updated_record = get_collection(tenant, collection_name).find_one_and_update(
            { '_id' : data.get('_id') },
            { '$set': data },
            new = True
        )
        updated_record = clean_object(updated_record)
        return updated_record

def delete(tenant, collection_name, search_criteria, user_id = None):
    search_criteria = declean_object(search_criteria)
    result = get_collection(tenant, collection_name).delete_many(search_criteria)
    return result

def clean_object(data):
    if data is not None and data.get('_id') is not None and type(data.get('_id')) == ObjectId:
        data['_id'] = str(data.get('_id'))
    return data

def clean_array(data):
    if data is not None and type(data) == list:
        for item in data:
            item = clean_object(item)
    return data

def declean_object(data):
    if data is not None and data.get('_id') is not None and type(data.get('_id')) != ObjectId:
        data['_id'] = ObjectId(data.get('_id'))
    return data

def declean_array(data):
    if data is not None and type(data) == list:
        for item in data:
            item = clean_object(item)
    return data
