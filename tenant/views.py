from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
import json

@api_view(['POST'])
def create(request):
    response = do_create(request.body)
    return JsonResponse(response[1], status=response[0])
