# coding=utf-8
from django.utils.deprecation import MiddlewareMixin
import jwt
import json
from django.core.exceptions import PermissionDenied

class JWTAuthenticationMiddleware(MiddlewareMixin):
    def process_request(self, request):

        if JWTAuthenticationMiddleware.is_json(request.body):
            request._body = json.loads(request.body)

        if request.path.startswith('/auth'):
            return

        claim = jwt.decode(request.headers.get('Authorization'), 'secret', algorithms=['HS256'])
        request.claim = claim

        if request.headers.get('Authorization') != '123':
            raise PermissionDenied
        
        return

    @staticmethod
    def is_json(data):
        try:
            json_object = json.loads(data)
        except ValueError as e:
            return False
        return True