from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('<str:id>', views.get),
    path('', views.do)
]