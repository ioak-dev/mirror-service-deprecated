import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from gridfs import GridFS
import base64
from bson.binary import Binary
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from library import db_utils
from app.stage.service import do_update_stages
from app.user.service import update_user

domain = 'mirror'
doamin_tenant = 'tenant'

def do_create(data_in, banner):
    data = data_in.copy()
    if banner != None:
        data['banner'] = banner.read()
    tenant = db_utils.upsert(domain, doamin_tenant, data)
    do_update_stages(data['name'],{'name':'Support'}, None)
    return (200, {'_id': tenant})

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

def do_update_tenant(tenant,data):
    tenantData = get_collection('mirror','tenant').find_one({'name': tenant})
    get_collection('mirror','tenant').update_one({
        '_id' : (tenantData['_id'])
    },{
        '$set':{
            'stage':data['data']
        }
    },upsert=True )
    return (200, None)
