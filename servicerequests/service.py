import os, datetime, time, random
from pymongo import mongo_client
from library.db_connection_factory import get_collection
from bson.objectid import ObjectId

DATABASE_URI = os.environ.get('DATABASE_URI')

def do_get_sr(tenant):
    existing_sr_list = get_collection(tenant, 'serviceRequests').find({})
    existing_sr = []
    for existing_srs in existing_sr_list:
        existing_srs['_id'] = str(existing_srs['_id'])
        existing_srs['createDate'] = str(existing_srs['createDate'])
        existing_srs['updateDate'] = str(existing_srs['updateDate'])
        existing_sr.append(existing_srs)
    print(existing_sr)
    return (200, {'data': existing_sr})

def do_add_update_sr(tenant, data):
    updated_id = 'id'
    stage_id ='stageId'
    if updated_id not in data and stage_id not in data :
        stage_list = list(get_collection(tenant, 'stage').find({}))
        #data['createDate'] = datetime.datetime.now().isoformat()
        #data['updateDate'] = datetime.datetime.now().isoformat()
        data['status'] = 'assigned'
        #data['priority'] = 'Low'
        data['stage'] = stage_list[0]['name']
        data['assignedTo'] = ''
        request = get_collection(tenant, 'serviceRequests').insert_one(data)
        return (200, {'_id': str(request.inserted_id)})
    elif updated_id not in data and stage_id in data :
        id = get_collection(tenant, 'serviceRequests').update_one({
            '_id' : ObjectId(data['stageId'])
        },{
            '$set':{
                'stage' : data['stage'],
                'previousStage' : data['previousStage'],
                'assignedTo': ''
            }
        }, upsert=True)
        return (200, {'_id': str(id)})
    else:
        id = get_collection(tenant, 'serviceRequests').update_one({
            '_id' : ObjectId(data['id'])
        },{
            '$set':{
                'title' : data['title'],
                'description' : data['description'],
                'priority' : data['priority'],
                'updateDate': data['updateTime'],
                'comments': data['comment'],
                'assignedTo': data['assignedTo']
            }
        }, upsert=True)
        return (200, {'_id': str(id)})