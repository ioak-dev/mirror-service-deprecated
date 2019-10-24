from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from auth.service import generate_keys, get_keys, do_signup, do_signin
from django.core import serializers
import json

@api_view(['GET'])
def keys(request, tenant):
    response = generate_keys()
    return JsonResponse(response[1], status=response[0])

@api_view(['POST'])
def signup(request, tenant):
    response = do_signup(tenant, request.body)
    return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def getKeys(request, tenant, email):
    response = get_keys(tenant, email)
    return JsonResponse(response[1], status=response[0])

@api_view(['POST'])
def signin(request, tenant):
    response = do_signin(tenant, request.body)
    return JsonResponse(response[1], status=response[0])