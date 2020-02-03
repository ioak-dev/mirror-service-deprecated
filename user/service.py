import os, datetime, time
from library.db_connection_factory import get_collection
import library.db_utils as db_utils

domain = 'user'

def find(request, tenant, id):
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