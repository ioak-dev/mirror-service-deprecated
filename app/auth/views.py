from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from app.auth.service import generate_keys, get_keys, do_signup, do_signin, do_jwttest, do_signin_via_jwt
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
    return HttpResponse(response[1].get('problem'), status=response[0])

@api_view(['POST'])
def signin(request, tenant):
    response = do_signin(tenant, request.body)
    return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def jwtTest(request, tenant):
    response = do_jwttest(tenant)
    return HttpResponse(response[1], status=response[0])

@api_view(['POST'])
def signin_jwt(request, tenant):
    response = do_signin_via_jwt(tenant, request.body)
    return JsonResponse(response[1], status=response[0])