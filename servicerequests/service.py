import os, datetime, time, random
from pymongo import mongo_client
from bson.objectid import ObjectId
import library.db_utils as db_utils

domain_sr_main="servicerequests.main"
domain_sr_log="servicerequests.log"

def get_sr_main(request, tenant):
    data = db_utils.find(tenant, domain_sr_main, {})
    return (200, {'data': data})

def update_sr_main(request, tenant):
    updated_record = db_utils.upsert(tenant, domain_sr_main, request.body, request.user_id)
    return (200, {'data': updated_record})

def get_sr_log(request, tenant, request_id):
    data = db_utils.find(tenant, domain_sr_log, {'request_id': request_id})
    return (200, {'data': data})

def update_sr_log(request, tenant, request_id):
    updated_record = db_utils.upsert(tenant, domain_sr_log, request.body, request.user_id)
    return (200, {'data': updated_record})
