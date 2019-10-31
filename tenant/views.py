from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
from tenant.service import do_create
from auth.service import do_signup
import json

@api_view(['POST'])
def create(request):
    response = do_create({
        'name': request.body.get('tenantName'),
        'ownerEmail': request.body.get('email'),
        'jwtPassword':request.body.get('email')
    })
    if response[0] == 200:
        response = do_signup(request.body.get('tenantName'), {
            'email': request.body.get('email'),
            'problem': request.body.get('problem'),
            'solution': request.body.get('solution')
        })
        return JsonResponse(response[1], status=response[0])
    else:
        return JsonResponse(response[1], status=response[0])
