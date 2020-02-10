from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import user.service as service

@api_view(['GET'])
def get(request, tenant, id):
    print(request.body, id)
    response = service.find(request, tenant, id)
    return JsonResponse(response[1], status=response[0])

@api_view(['GET', 'PUT'])
def do(request, tenant):
    if request.method == 'GET':        
        response = service.find_all(request, tenant)
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'PUT':
        response = service.update_user(request, tenant)
        return JsonResponse(response[1], status=response[0])
