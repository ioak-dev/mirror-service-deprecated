from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('dataset/<str:datatype>', views.dataset),
    path('model/<str:network_name>', views.model),
    path('model/<str:network_name>/train', views.train_model)
]