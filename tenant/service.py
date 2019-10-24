import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection

def do_create(data):
    tenant = get_collection('tenant').insert_one(data)
    return (200, {'_id': str(tenant.inserted_id)})
