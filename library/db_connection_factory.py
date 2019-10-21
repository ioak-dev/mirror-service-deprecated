from pymongo import MongoClient
import os

DATABASE_URI = os.environ.get('DATABASE_URI')
if DATABASE_URI is None:
    DATABASE_URI = 'mongodb://localhost:27017'

__connection_map = {}

def get_collection(tenant, collection):
    if tenant not in __connection_map.keys():
        __connection_map[tenant] = MongoClient(DATABASE_URI)[tenant]
    return __connection_map.get(tenant)[collection]