import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from gridfs import GridFS
import base64
from bson.binary import Binary
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

def do_add_stages(tenant, stages):
    stageData = get_collection(tenant,'stages').find_one({'name': tenant})
    if stageData == None:
        id = get_collection(tenant, 'stages').insert_one(stages)
        return (200, {'_id': str(id)})
    else:
        id = get_collection(tenant,'stages').update({
            '_id' : (stageData['_id'])
            },{
                '$set':{
                    'data':stages['stage']
                }
            }, upsert = True )
        return (200, {'_id': str(id)})
    
        

     