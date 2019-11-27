from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
from stage.service import do_add_stage
import json, base64

@api_view(['PUT'])
def add_stage(request,tenant):
    response = do_add_stage(tenant, request.body)
    return JsonResponse(response[1], status=response[0])
