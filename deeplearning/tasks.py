from mirror.celery import app
from library.db_connection_factory import get_collection
from deeplearning.models import TransientModel, ModelContainer
from library.collection_utils import list_to_dict
import tensorflow as tf
import pandas as pd
# from celery.decorators import shared_task

# logger=get_task_logger(__name__)

# This is the decorator which a celery worker uses
# @shared_task(name="test_task")
@app.task(bind=True)
def train_model(self, tenant):
    model = TransientModel(tenant)
    # model.initialize_vectorizer(pd.DataFrame(list(get_collection(tenant, 'dataset_train').find({}))))
    print('initializing vectorizer')
    model.initialize_vectorizer()
    print('initialized vectorizer')
    categories = get_collection(tenant, 'category').find({})
    label_map = list_to_dict(list(categories), 'name', 'value')
    model.train(tenant, label_map, pd.DataFrame(list(get_collection(tenant, 'dataset_train').find({}))), pd.DataFrame(list(get_collection(tenant, 'dataset_test').find({}))))
    return {'label_count': 'label_count'}
