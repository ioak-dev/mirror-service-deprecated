from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('dataset', views.dataset),
    path('model', views.model),
    path('model/train', views.train_model),
    path('model/load', views.load_model),
    path('labels/load', views.load_labels),
    path('model/predict', views.predict)
]