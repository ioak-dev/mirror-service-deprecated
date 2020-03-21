#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
from library.db_connection_factory import get_collection
from app.learning import service
from app.learning import models

def run():
    tenants = service.get_all_tenants()
    for tenant in tenants:
        print('Train model for tenant: ' + tenant['name'])
        service.train_model(tenant['name'])
        print('Load model for tenant: ' + tenant['name'])
        service.load_model(tenant['name'])
    
    



