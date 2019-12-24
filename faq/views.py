from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import faq.service as service

@api_view(['GET', 'POST', 'PUT'])
def get_update_faq(request, tenant):
    if request.method == 'POST':
        response = service.do_add_faq(tenant, request.body)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'GET':
        response = service.do_get_faq(tenant)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT':
        response = service.do_add_faq(tenant, request.body)
        return JsonResponse(response[1], status=response[0])
    
