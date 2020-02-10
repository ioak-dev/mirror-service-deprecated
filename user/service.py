import os, datetime, time
from library.db_connection_factory import get_collection
import library.db_utils as db_utils

domain = 'user'
domain_role_permissions = 'role_permissions'

def find(tenant, id):
    data = db_utils.find(tenant, domain, {'_id': id})
    return (200, {'data': data})

def find_all(request, tenant):
    data = db_utils.find(tenant, domain, {})
    return (200, {'data': data})

def expand_authors(tenant, data):
    for item in data:
        last_modified_by = db_utils.find(tenant, domain, {'_id': item.get('lastModifiedBy')})
        created_by = db_utils.find(tenant, domain, {'_id': item.get('createdBy')})
        item['lastModifiedByEmail'] = last_modified_by[0].get('email')
        item['createdByEmail'] = created_by[0].get('email')
    return data

def update_user(request, tenant):
    print(request.body)
    updated_record = db_utils.upsert(tenant, domain, request.body, request.user_id)
    return (200, {'data': updated_record})


def find_permitted_actions(tenant, user_id):
    roles = db_utils.find(tenant, domain, {'_id': user_id})[0].get('role')
    return db_utils.find(tenant, domain_role_permissions, {'role': {'$in': roles}})

def can_i_perform(permitted_actions, action, domain, condition, group=None):
    for item in permitted_actions:
        if item.get('action') == action and item.get('domain') == domain and item.get('condition') == condition:
            if group == None or item.get('group') == group:
                return True
    return False

def who_can_perform(permitted_actions, action, domain, condition):
    group_list = []
    for item in permitted_actions:
        print(item)
        if item.get('action') == action and item.get('domain') == domain and item.get('condition') == condition and item.get('group') != None:
                group_list.append(item.get('group'))
    return group_list
