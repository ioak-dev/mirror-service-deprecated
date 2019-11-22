def list_to_dict(data_in, key, value):
    data_out = {}
    for entry in data_in:
        data_out[entry[key]] = entry[value]
    return data_out