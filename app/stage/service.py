import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import library.db_utils as db_utils

domain = 'stage'

def do_get_stages(tenant):
    existing_stage = db_utils.find(tenant, domain,{})
    return (200, {'stage':existing_stage})

def do_update_stages(tenant, request):
    #get_collection(tenant, 'stage').remove()
    db_utils.delete(tenant, domain, {})
    #updated_record = db_utils.upsert(tenant, domain,request.body, request.user_id)
    local_stage = get_collection(tenant, 'stage').insert_many(request.body)
    return (200, {'_id': str(local_stage.inserted_ids)})
    #return (200, {'data':updated_record})
    
