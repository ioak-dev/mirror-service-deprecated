instances = {}

def add(tenant, network_name, model):
    instances[tenant] = {network_name: model}

def get(tenant, network_name):
    if tenant in instances and network_name in instances[tenant]:
        print('PRESENT****')
        return instances[tenant][network_name]
    else:
        print('ABSENT****')

def remove(tenant, network_name):
    if tenant in instances and network_name in instances[tenant]:
        del instances[tenant][network_name]

    