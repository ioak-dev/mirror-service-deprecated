from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
from tenant.service import do_create
import json

@api_view(['POST'])
def create(request,tenant):
    response = do_create(tenant, request.body)
    return JsonResponse(response[1], status=response[0])
