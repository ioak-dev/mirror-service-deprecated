import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection

DATABASE_URI = os.environ.get('DATABASE_URI')

def do_get_faq(tenant):
    existing_faq_list = get_collection(tenant, 'Faq').find()
    existing_faq = []
    for existing_faqs in existing_faq_list:
        existing_faqs['_id'] = str(existing_faqs['_id'])
        existing_faq.append(existing_faqs)
    return (200, {'faq': existing_faq})

def do_add_faq(tenant, data):
    id = get_collection(tenant, 'Faq').insert_one(data)
    return (200, {'_id': str(id)})