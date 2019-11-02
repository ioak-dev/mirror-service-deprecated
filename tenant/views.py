from django.http import HttpResponse, JsonResponse
from rest_framework.decorators import api_view
from django.core import serializers
from tenant.service import do_create, do_get_tenant, do_get_banner
from auth.service import do_signup
import json, base64

@api_view(['POST'])
def create(request):
    if request.FILES != None:
        banner = request.FILES.get('banner')
    else:
        banner = None
    response = do_create({
        'name': request.POST.get('tenantName'),
        'ownerEmail': request.POST.get('email'),
        'jwtPassword':request.POST.get('jwtPassword')
    }, banner)
    if response[0] == 200:
        response = do_signup(request.POST.get('tenantName'), {
            'email': request.POST.get('email'),
            'problem': request.POST.get('problem'),
            'solution': request.POST.get('solution')
        })
        return JsonResponse(response[1], status=response[0])
    else:
        return JsonResponse(response[1], status=response[0])

@api_view(['GET'])
def get_banner(request, tenant):
    response = do_get_banner(tenant)
    return HttpResponse(response[1], status=response[0])

@api_view(['GET'])
def get_tenant(request, tenant):
    response = do_get_tenant(tenant)
    return JsonResponse(response[1], status=response[0])
