from mirror.celery import app
from library.db_connection_factory import get_collection
from deeplearning.models import Model, ModelContainer
from library.collection_utils import list_to_dict
import tensorflow as tf
# from celery.decorators import shared_task

# logger=get_task_logger(__name__)

# This is the decorator which a celery worker uses
# @shared_task(name="test_task")
@app.task(bind=True)
def vectorize(self, tenant, network_name):
    model = Model(network_name)
    ModelContainer.add(tenant, network_name, model)
    model = ModelContainer.get(tenant, network_name)
    # model.initialize_vectorizer(pd.DataFrame(list(get_collection(tenant, 'dataset_train').find({}))))
    print('initializing vectorizer')
    model.initialize_vectorizer()
    print('initialized vectorizer')
    categories = get_collection(tenant, 'category').find({})
    label_map = list_to_dict(list(categories), 'name', 'value')
    write_to_tfrecord(tenant, 'val', label_map, model.vectorizer)
    label_count = write_to_tfrecord(tenant, 'train', label_map, model.vectorizer)
    write_to_tfrecord(tenant, 'test', label_map, model.vectorizer)
    model.label_count = label_count
    return {'label_count': label_count}

def write_to_tfrecord(tenant, datatype, label_map, vectorizer):
    print('vectorizing', datatype)
    label_count = set()
    cursor = get_collection(tenant, 'dataset_' + datatype).find({})
    with tf.io.TFRecordWriter('data/' + tenant + '_' + datatype + '.tfrecords') as writer:
        i = 0
        for item in cursor:
            i = i + 1
            features = vectorizer.transform([item['text']]).toarray()[0]
            label = label_map.get(item['label'])
            # print(vector[0].shape)
            # print(label)
            label_count.add(label)
            example = tf.train.Example()
            example.features.feature["features"].float_list.value.extend(features)
            example.features.feature["label"].int64_list.value.append(label)
            writer.write(example.SerializeToString())
    print('label_count', datatype, len(label_count))
    print('record_count', datatype, i)
    return len(label_count)

@app.task(bind=True)
def train_model(self, tenant, network_name):
    model = ModelContainer.get(tenant, network_name)
    if model == None:
        return (204, {'error': 'model not present uner network name [' + network_name + ']'})
    else:
        model.train(tenant)
        return (200, {})
# @app.task(bind=True)
# def test_task(self, job_name=None):

#     b = Tasks(task_id=self.request.id, job_name=job_name)
#     b.save()

#     self.update_state(state='Dispatching', meta={'progress': '33'})
#     sleep(random.randint(5, 10)) 

#     self.update_state(state='Running', meta={'progress': '66'})
#     sleep(random.randint(5, 10))  
#     self.update_state(state='Finishing', meta={'progress': '100'})
#     sleep(random.randint(5, 10)) 