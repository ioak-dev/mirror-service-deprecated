import os, datetime, time, random
from pymongo import mongo_client
from bson.objectid import ObjectId
import library.db_utils as db_utils

domain=["serviceRequests","stage"]

def do_get_sr(tenant):
    existing_sr_list = db_utils.find(tenant, domain[0], {})
    return (200, {'data': existing_sr_list})


def do_add_update_sr(tenant, data):
    stage_list = db_utils.find(tenant, domain[1],{})
    data['stage'] = stage_list[0]['name']
    updated_record = db_utils.upsert(tenant, domain[0], data)
    return (200, {'data': updated_record})
