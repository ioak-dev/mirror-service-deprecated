import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from gridfs import GridFS
import base64
from bson.binary import Binary
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import json

#get existing stages to display
def do_get_stages(tenant):
    existing_stage_list = get_collection(tenant,'stage').find()
    existing_stage = []
    for existing_stages in existing_stage_list:
        existing_stages['_id'] = str(existing_stages['_id'])
        existing_stage.append(existing_stages)
    return (200, (existing_stage))

# update or remove or insert stages
def do_update_stages(tenant, list_stage_latest):
    get_collection(tenant, 'stage').remove()
    local_stage = get_collection(tenant, 'stage').insert_many(list_stage_latest)
    return (200, {'_id': str(local_stage.inserted_ids)})
    
