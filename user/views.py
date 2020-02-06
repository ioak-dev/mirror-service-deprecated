from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import user.service as service

@api_view(['GET','PUT'])
def get(request, tenant, id):
    if request.method == 'GET' :
        print(request.body, id)
        response = service.find(request, tenant, id)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT' :
        print(request.body, id)
        response = service.update_user(request, tenant, id)
        return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def get_all(request, tenant):
    response = service.find_all(request, tenant)
    return JsonResponse(response[1], status=response[0])
