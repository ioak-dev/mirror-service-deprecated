from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('', views.get_update_stages),
    path('<str:id>', views.delete_stage)
]