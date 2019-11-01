import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from gridfs import GridFS
import base64
from bson.binary import Binary
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

DATABASE_URI = os.environ.get('DATABASE_URI')

def do_create(data_in, banner):
    data = data_in.copy()
    if banner != None:
        data['banner'] = banner.read()
    tenant = get_collection('mirror','tenant').insert_one(data)
    return (200, {'_id': str(tenant.inserted_id)})

def do_get_banner(tenant):
    tenantData = get_collection('mirror','tenant').find_one({'name': tenant})
    tenantData['_id'] = str(tenantData['_id'])
    if 'banner' in tenantData:
        return (200, base64.b64encode(tenantData['banner']))
    else:
        return (404, None)


def do_get_tenant(tenant):
    tenantData = get_collection('mirror','tenant').find_one({'name': tenant})
    tenantData['_id'] = str(tenantData['_id'])
    tenantData.pop('banner', None)
    return (200, tenantData)

