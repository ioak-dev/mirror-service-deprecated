import os, datetime, time
from pymongo import MongoClient
import secrets, jwt
from library.db_connection_factory import get_collection

DATABASE_URI = os.environ.get('DATABASE_URI')

def generate_keys():
    return (200, {
        'salt': secrets.token_hex(40),
        'solution': secrets.token_hex(40)
    })

def get_keys(tenant, email):
    user = get_collection(tenant, 'user').find_one({'email': email})
    if user is None:
        return (404, {})
    else:
        return (200, {'problem': user.get('problem')})

def do_signup(tenant, data):
    user = get_collection(tenant, 'user').insert_one(data)
    return (200, {'_id': str(user.inserted_id)})

def do_signin(tenant, data):
    user = get_collection(tenant, 'user').find_one({'email': data.get('email')})
    response = {'content': {}}
    if user is None:
        return (404, {})
    elif user.get('solution') != data.get('solution'):
        return (401, {})
    elif user.get('solution') == data.get('solution'):
        return (200, {
            'name': user.get('name'),
            'email': user.get('email'),
            'token': jwt.encode({
                'userId': str(user.get('_id')),
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)
                }, 'jwtsecret').decode('utf-8'),
            'tenant': tenant,
            'secret': 'none'
        })

def jwtTest():
    en = jwt.encode({'some': 'payload', 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=3)}, 'secret', algorithm='HS256')
    print(jwt.decode(en, 'secret', algorithms=['HS256']))
    # time.sleep(10)
    # print(jwt.decode(en, 'secret', algorithms=['HS256'], verify=False))
    # print(jwt.decode(en, 'secret', algorithms=['HS256']))
    return jwt.encode({'some': 'payload', 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=30)}, 'secret', algorithm='HS256')
