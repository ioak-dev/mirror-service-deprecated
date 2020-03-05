from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import app.stage.service as service
#from stage.service import do_update_stages, do_get_stages
import json, base64

@api_view(['PUT','GET'])
def get_update_stages(request,tenant):
    if request.method == 'GET':
        response = service.do_get_stages(tenant)
        return JsonResponse(response[1], status=response[0])
    if request.method == 'PUT':
        response = service.do_update_stages(tenant, request.body, request.user_id)
        return JsonResponse(response[1], status=response[0])
    

@api_view(['DELETE'])
def delete_stage(request, id, tenant):
    response = service.do_delete_stage(request, tenant, id)
    return JsonResponse(response[1], status=response[0])