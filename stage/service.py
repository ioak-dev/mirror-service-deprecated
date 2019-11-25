import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from gridfs import GridFS
import base64
from bson.binary import Binary
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

def do_add_stage(tenant, stage):
    stage = stage['stage']
    for i in stage:
        stage_data = get_collection(tenant,'stage').find_one({'name': i['name']})
        if stage_data == None:
            id = get_collection(tenant, 'stage').insert_one(i)
            return (200, {'_id': str(id)})
    return (200, {'_id': None})