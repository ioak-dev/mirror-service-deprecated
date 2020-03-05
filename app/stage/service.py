import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
import library.db_utils as db_utils

domain = 'stage'

def do_get_stages(tenant):
    existing_stage = db_utils.find(tenant, domain,{})
    return (200, {'stage':existing_stage})

def do_update_stages(tenant, data, user_id):
    updated_record = db_utils.upsert(tenant, domain,data, user_id)
    return (200, {'data':updated_record})

def do_delete_stage(request, tenant, id):
    result = db_utils.delete(tenant, domain, {'_id': id}, request.user_id)
    return (200, {'deleted_count': result.deleted_count})