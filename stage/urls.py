from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('', views.add_stage),
    path('remove', views.remove_stage)
]