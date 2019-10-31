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

def do_jwttest(tenant):
    tenant=get_collection('mirror', 'tenant').find_one({'name': tenant})
    jwtPassword = tenant.get('jwtPassword')
    return (200, jwt.encode({
            'userId': '4587439657496t',
            'name': 'test user display name',
            'email': 'q1@1.com',
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)
        }, jwtPassword, algorithm='HS256').decode('utf-8'))

def do_signin_via_jwt(tenant, data):
    tenantData=get_collection('mirror', 'tenant').find_one({'name': tenant})
    jwtPassword = tenantData.get('jwtPassword')
    jwtToken = data.get('jwtToken')
    tokenData = jwt.decode(jwtToken, jwtPassword, algorithm='HS256')
    user = get_collection(tenant, 'user').find_one({'email': tokenData.get('email')})
    if user is None:
        get_collection(tenant, 'user').insert_one({
            'name': tokenData.get('name'),
            'email': tokenData.get('email'),
            'type': 'JWT_USER'
        })
    else:
        get_collection(tenant, 'user').update({'_id': user.get('_id')},
        {
            'name': tokenData.get('name'),
            'email': tokenData.get('email'),
            'type': 'JWT_USER'
        }, upsert=True)
    
    user = get_collection(tenant, 'user').find_one({'email': tokenData.get('email')})
    return (200, {
        'name': user.get('name'),
        'email': user.get('email'),
        'token': jwt.encode({
                'name': str(user.get('_id')),
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=8)
            }, jwtPassword, algorithm='HS256').decode('utf-8'),
        'tenant': tenant,
        'secret': 'none'
    })

    # print(jwt.decode(en, 'secret', algorithms=['HS256']))
    # time.sleep(10)
    # print(jwt.decode(en, 'secret', algorithms=['HS256'], verify=False))
    # print(jwt.decode(en, 'secret', algorithms=['HS256']))
    # return jwt.encode({'some': 'payload', 'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=30)}, 'secret', algorithm='HS256')
