from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('main', views.sr_main),
    path('log', views.sr_log)
]