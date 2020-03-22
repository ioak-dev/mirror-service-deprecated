import os, datetime, time, random
from pymongo import mongo_client
from bson.objectid import ObjectId
import library.db_utils as db_utils
from app.user.service import expand_authors, find_permitted_actions, can_i_perform, who_can_perform

domain = "servicerequests"
domain_sr_main="servicerequests.main"
domain_sr_log="servicerequests.log"

def get_sr_main(request, tenant):
    permitted_actions = find_permitted_actions(tenant, request.user_id)
    condition = None
    if can_i_perform(permitted_actions, 'read', domain, "all"):
        condition = {}
    elif can_i_perform(permitted_actions, 'read', domain, "assigned to group"):
        condition = {'$or': [{'assignedTo': {'$in': who_can_perform(permitted_actions, 'read', domain, "assigned to group")}}, {'createdBy': request.user_id}]}
    elif can_i_perform(permitted_actions, 'read', domain, "created by me"):
        condition = {'createdBy': request.user_id}
    data = db_utils.find(tenant, domain_sr_main, condition)
    return (200, {'data': data})

def update_sr_main(request, tenant):
    updated_record = db_utils.upsert(tenant, domain_sr_main, request.body, request.user_id)
    return (200, {'data': updated_record})

def get_sr_log(request, tenant, request_id):
    data = expand_authors(tenant, db_utils.find(tenant, domain_sr_log, {'request_id': request_id}))
    return (200, {'data': data})

def update_sr_log(request, tenant, request_id):
    updated_record = db_utils.upsert(tenant, domain_sr_log, request.body, request.user_id)
    return (200, {'data': updated_record})
