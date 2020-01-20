import os, datetime, time, random
from pymongo import mongo_client
from bson.objectid import ObjectId
import library.db_utils as db_utils

domain=["serviceRequests","stage"]

def do_get_sr(request, tenant):
    data = db_utils.find(tenant, domain[0], {})
    return (200, {'data': data})


def do_add_update_sr(request, tenant):
    stage_list = db_utils.find(tenant, domain[1],{})
    data['stage'] = stage_list[0]['name']
    updated_record = db_utils.upsert(tenant, domain[0], request.body, request.user_id)
    return (200, {'data': updated_record})
