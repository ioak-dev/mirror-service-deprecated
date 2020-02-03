from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import faq.service as service

@api_view(['GET', 'PUT'])
def get_update_faq(request, tenant):
    if request.method == 'GET':
        response = service.find(request, tenant)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT':
        response = service.update(request, tenant, request.body)
        return JsonResponse(response[1], status=response[0])
    
@api_view(['DELETE'])
def delete_faq(request,tenant,id):
    if request.method == 'DELETE':
        response = service.delete(request, tenant, id)
        return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def get_by_category(request, tenant, category):
    if request.method == 'GET':
        response = service.find_faq_by_category(request, tenant, category)
        return JsonResponse(response[1], status=response[0])