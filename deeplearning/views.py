from csv import reader
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
import deeplearning.service as service

# Create your views here.

@api_view(['GET', 'POST', 'DELETE'])
def dataset(request, tenant):
    if request.method == 'GET':
        response = service.get_dataset(tenant, request.body.decode('utf-8'))
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'POST':
        response = service.add_dataset(tenant, request.body.decode('utf-8'))
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'DELETE':
        response = service.clear_dataset(tenant, request.body.decode('utf-8'))
        return JsonResponse(response[1], status=response[0])


@api_view(['POST', 'DELETE'])
def model(request, tenant, network_name):
    if request.method == 'POST':
        response = service.create_model(tenant, network_name)
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'DELETE':
        response = service.remove_model(tenant, network_name)
        return JsonResponse(response[1], status=response[0])

@api_view(['POST'])
def train_model(request, tenant, network_name):
    response = service.train_model(tenant, network_name)
    return JsonResponse(response[1], status=response[0])

@api_view(['POST'])
def featuretext_to_vector(request, tenant, network_name):
    response = service.featuretext_to_vector(tenant, network_name)
    return JsonResponse(response[1], status=response[0])

@api_view(['POST'])
def predict(request, tenant, network_name):
    response = service.predict(tenant, network_name, request.body.decode('utf-8'))
    return JsonResponse(response[1], status=response[0])
