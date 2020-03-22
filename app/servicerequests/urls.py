from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('main', views.sr_main),
    path('log/<str:request_id>', views.sr_log)
]