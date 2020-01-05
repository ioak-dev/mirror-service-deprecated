import os, datetime, time
from pymongo import mongo_client
from library.db_connection_factory import get_collection

DATABASE_URI = os.environ.get('DATABASE_URI')

def do_get_sr(tenant):
    existing_sr_list = get_collection(tenant, 'serviceRequests').find({})
    existing_sr = []
    for existing_srs in existing_sr_list:
        existing_srs['_id'] = str(existing_srs['_id'])
        existing_srs['createDate'] = str(existing_srs['createDate'])
        existing_srs['updateDate'] = str(existing_srs['updateDate'])
        existing_sr.append(existing_srs)
    return (200, {'data': existing_sr})

def do_add_update_sr(tenant, data):
    existing_req_no = list(get_collection(tenant, 'serviceRequests').find({'requestNo':{"$exists": True}} ,sort = [('requestNo',-1)]).limit(1))
    if not existing_req_no:
        requestNo = 'ioak1'
    else :
        temp_request_number = int(existing_req_no[0]['requestNo'][4:])+1
        requestNo = 'ioak'+str(temp_request_number)
    data['requestNo'] = requestNo
    data['createDate'] = datetime.datetime.now().isoformat()
    data['updateDate'] = datetime.datetime.now().isoformat()
    data['status'] = 'open'
    id = get_collection(tenant, 'serviceRequests').insert_one(data)
    return (200, {'_id': str(id)})