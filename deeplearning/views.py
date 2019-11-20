from csv import reader
from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from deeplearning.service import get_dataset, add_dataset, clear_dataset

# Create your views here.

@api_view(['GET', 'POST', 'DELETE'])
def dataset(request, tenant, datatype):
    if request.method == 'GET':
        response = get_dataset(tenant, request.body.decode('utf-8'), datatype)
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'POST':
        response = add_dataset(tenant, request.body.decode('utf-8'), datatype)
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'DELETE':
        response = clear_dataset(tenant, request.body.decode('utf-8'), datatype)
        return JsonResponse(response[1], status=response[0])
