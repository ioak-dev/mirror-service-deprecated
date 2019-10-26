import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection

DATABASE_URI = os.environ.get('DATABASE_URI')

def do_create(data):
    tenant = get_collection('mirror','tenant').insert_one(data)
    return (200, {'_id': str(tenant.inserted_id)})

