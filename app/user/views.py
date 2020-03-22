from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import app.user.service as service

@api_view(['GET', 'PUT'])
def do(request, tenant):
    if request.method == 'GET':
        response = service.find(request, tenant)
        return JsonResponse(response[1], status=response[0])
    elif request.method == 'PUT':
        response = service.update_user(request, tenant)
        return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def get_all(request, tenant):
    if request.method == 'GET':
        response = service.find_all(request, tenant)
        return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def permittedActions(request, tenant):
    response = service.find_permitted_actions(tenant, request.user_id)
    return (200, {'data': response})
