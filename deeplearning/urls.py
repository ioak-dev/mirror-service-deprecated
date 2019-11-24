from django.urls import path
from django.conf.urls import url

from . import views

urlpatterns =[
    path('dataset', views.dataset),
    path('model/<str:network_name>', views.model),
    path('model/<str:network_name>/vectorize', views.featuretext_to_vector),
    path('model/<str:network_name>/train', views.train_model),
    path('model/<str:network_name>/predict', views.predict)
]