from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('create', views.create),
    path('<str:tenant>', views.get_tenant),
    path('banner/<str:tenant>', views.get_banner),
    path('stage/<str:tenant>', views.add_stage)
]