from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import servicerequests.service as service

@api_view(['GET', 'PUT'])
def get_update_sr(request, tenant):
    if request.method == 'GET':
        response = service.do_get_sr(request, tenant)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT':
        response = service.do_add_update_sr(request, tenant)
        return JsonResponse(response[1], status=response[0])