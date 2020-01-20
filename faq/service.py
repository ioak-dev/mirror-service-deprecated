import os, datetime, time
from library.db_connection_factory import get_collection
import library.db_utils as db_utils

domain = 'Faq'

def find(tenant):
    faq_list = db_utils.find(tenant, domain, {})
    category_list = []
    for item in faq_list:
        if item.get('category') not in category_list:
            category_list.append(item['category'])
    return (200, {'faq': faq_list ,'category': category_list})

def update(tenant, data):
    updated_record = db_utils.upsert(tenant, domain, data)
    return (200, {'data': updated_record})

def delete(tenant, id):
    result = db_utils.delete(tenant, domain, {'_id': id})
    return (200, {'deleted_count': result.deleted_count})

def find_faq_by_category(tenant, category):
    data = db_utils.find(tenant, domain, {'category': category})
    return (200, {'data': data})
