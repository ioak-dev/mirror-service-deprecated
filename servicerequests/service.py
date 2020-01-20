import os, datetime, time, random
from pymongo import mongo_client
from bson.objectid import ObjectId
import library.db_utils as db_utils

domain="serviceRequests"

def do_get_sr(request, tenant):
    data = db_utils.find(tenant, domain, {})
    return (200, {'data': data})


def do_add_update_sr(request, tenant):
    updated_record = db_utils.upsert(tenant, domain, request.body, request.user_id)
    return (200, {'data': updated_record})
