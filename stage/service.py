import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from gridfs import GridFS
import base64
from bson.binary import Binary
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

def do_add_stage(tenant, stage):
    all_stages = get_collection(tenant,'stage').find({}, {'_id': False})
    id = 0
    for j in all_stages :
        for i in stage :
            if j['name'] not in i :
                id = get_collection(tenant, 'stage').remove({'name' : j["name"]})
    for i in stage :
        tempStage = get_collection(tenant,'stage').find_one({'name': i['name']})
        if tempStage is None :
            id = get_collection(tenant, 'stage').insert_one(i)
            
    return (200, {'_id': str(id)})