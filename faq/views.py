from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import faq.service as service

@api_view(['GET', 'PUT'])
def get_update_faq(request, tenant):
    if request.method == 'GET':
        response = service.do_get_faq(tenant)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT':
        response = service.do_add_update_faq(tenant, request.body)
        return JsonResponse(response[1], status=response[0])
    
    
@api_view(['DELETE'])
def delete_faq(request,tenant,id):
    if request.method == 'DELETE':
        response = service.do_delete_faq(tenant, id)
        return JsonResponse(response[1], status=response[0])
