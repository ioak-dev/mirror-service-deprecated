import os, datetime, time
from pymongo import MongoClient
from library.db_connection_factory import get_collection
from bson.objectid import ObjectId

DATABASE_URI = os.environ.get('DATABASE_URI')

def do_get_faq(tenant):
    existing_faq_list = get_collection(tenant, 'Faq').find()
    existing_faq = []
    existing_categories = ([])
    for existing_faqs in existing_faq_list:
        existing_faqs['_id'] = str(existing_faqs['_id'])
        existing_faq.append(existing_faqs)
        if existing_faqs['category'] not in existing_categories:
            existing_categories.append(existing_faqs['category'])
    print(existing_categories)
    return (200, {'faq': existing_faq ,'category': existing_categories})

def do_add_update_faq(tenant, data):
    updated_id = 'id'
    if updated_id not in data:
        id = get_collection(tenant, 'Faq').insert_one(data)
        return (200, {'_id': str(id)})
    else:
        id = get_collection(tenant, 'Faq').update_one({
            '_id' : ObjectId(data['id'])
        },{
            '$set':{
                'category' : data['category'],
                'question' : data['question'],
                'answer' : data['answer']
            }
        }, upsert=True)
        return (200, {'_id': str(id)})

def do_delete_faq(tenant, id):
    result = get_collection(tenant,'Faq').delete_one({'_id': ObjectId(id)})
    return (200, {'deleted_count':result.deleted_count})