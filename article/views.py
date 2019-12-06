from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import article.service as service
import json, base64

@api_view(['GET','POST','PUT'])
def category(request, tenant):
    if request.method == 'POST':
        response = service.add_category(tenant, request.body)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'GET':
        response = service.get_category_all(tenant)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT':
        response = service.add_category(tenant,request.body)
        return JsonResponse(response[1], status=response[0])