#authenticatoion/views.py
from datetime import datetime, timedelta
from django.utils import timezone
import random
import string
import logging
from rest_framework import generics, status, views
from rest_framework.response import Response
from .models import User
from .serializers import (
    RegisterSerializer,
    EmailVerificationSerializer,
    LoginSerializer,
    RequestPasswordResetEmailSerializer,
    SetNewPasswordWithOTPSerializer
)
from .utils import Util, generate_otp, google_authenticate, is_otp_valid
from camera.models import CameraStream
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.core.cache import cache

logger = logging.getLogger(__name__)

class GoogleSignInView(views.APIView):
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'token': openapi.Schema(type=openapi.TYPE_STRING, description='Google token'),
            },
            required=['token'],
        ),
        responses={
            200: openapi.Response('Success', openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'user_info': openapi.Schema(
                        type=openapi.TYPE_OBJECT,
                        properties={
                            'username': openapi.Schema(type=openapi.TYPE_STRING),
                            'email': openapi.Schema(type=openapi.TYPE_STRING),
                        }
                    ),
                    'access_token': openapi.Schema(type=openapi.TYPE_STRING),
                    'refresh_token': openapi.Schema(type=openapi.TYPE_STRING),
                    'stream_urls': openapi.Schema(type=openapi.TYPE_ARRAY, items=openapi.Items(type=openapi.TYPE_STRING)),
                }
            )),
            400: 'Invalid token or missing parameter',
            500: 'Internal server error'
        }
    )
    def post(self, request):
        token = request.data.get('token')
        if not token:
            return Response({'error': 'Token is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        user = google_authenticate(token)
        if user is None:
            return Response({'error': 'Invalid token'}, status=status.HTTP_400_BAD_REQUEST)
        elif user == 'existing_email':
            return Response({'error': 'This email is already associated with a regular account'}, status=status.HTTP_400_BAD_REQUEST)

        # Create tokens for the user
        tokens = user.tokens()
        
        # Retrieve all stream URLs for the user
        streams = CameraStream.objects.filter(user=user)
        stream_urls = [stream.stream_url for stream in streams]

        return Response({
            'user_info': {
                'username': user.username,
                'email': user.email,
            },
            'access_token': tokens['access'],
            'refresh_token': tokens['refresh'],
            'stream_urls': stream_urls,
        }, status=status.HTTP_200_OK)



class RegisterView(generics.GenericAPIView):
    serializer_class = RegisterSerializer

    def post(self, request):
        user_data = request.data.copy()  # Create a mutable copy of the request data
        serializer = self.serializer_class(data=user_data)
        serializer.is_valid(raise_exception=True)

        email = user_data.get('email')
        if User.objects.filter(email=email).exists():
            return Response({'error': 'Email is already registered'}, status=status.HTTP_400_BAD_REQUEST)

        verification_code = ''.join(random.choices(string.digits, k=6))
        user_data['verification_code'] = verification_code
        user_data['verification_code_expires_at'] = (timezone.now() + timedelta(minutes=10)).isoformat()

        try:
            cache.set(verification_code, user_data, timeout=600)  # Cache for 10 minutes

            email_body = f'Hi {user_data["username"]}, use the verification code below to verify your email address:\n{verification_code}'
            data = {
                'email_body': email_body,
                'to_email': user_data['email'],
                'email_subject': 'Verify Your Email'
            }

            Util.send_email(data)
        except Exception as e:
            logger.error(f"Error in registration process: {e}")
            return Response({'error': 'An error occurred while sending the verification email. Please try again later.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({'detail': 'Verification code sent to your email'}, status=status.HTTP_201_CREATED)



class VerifyEmail(views.APIView):
    serializer_class = EmailVerificationSerializer

    @swagger_auto_schema(
        request_body=EmailVerificationSerializer,
        responses={
            200: openapi.Response('Email successfully verified', EmailVerificationSerializer),
            400: 'Invalid or expired verification code',
            500: 'Internal server error'
        }
    )
    def post(self, request):
        logger.debug('Received request data: %s', request.data)
        
        serializer = self.serializer_class(data=request.data)
        if not serializer.is_valid():
            logger.error('Validation errors: %s', serializer.errors)
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        code = serializer.validated_data.get('code')
        user_data = cache.get(code)

        if not user_data:
            logger.error('Invalid or expired verification code: %s', code)
            return Response({'error': 'Invalid or expired verification code'}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure both datetimes are timezone-aware
        verification_code_expires_at = datetime.fromisoformat(user_data['verification_code_expires_at'])
        if verification_code_expires_at.tzinfo is None:
            verification_code_expires_at = timezone.make_aware(verification_code_expires_at, timezone.utc)

        now = timezone.now()

        if verification_code_expires_at < now:
            cache.delete(code)  # Clear the expired cached data
            logger.error('Verification code has expired: %s', code)
            return Response({'error': 'Verification code has expired'}, status=status.HTTP_400_BAD_REQUEST)

        # Remove sensitive fields before saving
        user_data.pop('verification_code')
        user_data.pop('verification_code_expires_at')

        user = User(
            email=user_data['email'],
            username=user_data['username']
        )
        user.set_password(user_data['password'])
        user.is_active = True  # Activate user
        user.is_verified = True  # Mark user as verified
        user.save()
        cache.delete(code)  # Clear the cached data

        email_body = f'Hi {user.username}, your email has been successfully verified. You can now log in.'
        data = {
            'email_body': email_body,
            'to_email': user.email,
            'email_subject': 'Email Verified'
        }
        Util.send_email(data)

        return Response({'detail': 'Email successfully verified'}, status=status.HTTP_200_OK)


class LoginAPIView(generics.GenericAPIView):
    serializer_class = LoginSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = serializer.validated_data['user']

        # Retrieve all stream URLs for the user
        streams = CameraStream.objects.filter(user=user)
        stream_urls = [{'id': stream.id, 'name': stream.camera.name, 'url': f'ws://13.200.111.211/ws/camera/{stream.id}/'} for stream in streams]

        return Response({
            'user_info': {
                'username': user.username,
                'email': user.email,
            },
            'access_token': str(serializer.validated_data['tokens']['access']),
            'refresh_token': str(serializer.validated_data['tokens']['refresh']),
            'stream_urls': stream_urls,
        }, status=status.HTTP_200_OK)


class RequestPasswordResetEmail(generics.GenericAPIView):
    serializer_class = RequestPasswordResetEmailSerializer

    def post(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)

        email = serializer.validated_data['email']
        user = User.objects.get(email=email)
        
        otp_request_count = cache.get(f'otp_requests_{email}', 0)
        first_request_time = cache.get(f'first_otp_request_{email}', None)
        
        if first_request_time and timezone.now() > first_request_time + timedelta(hours=1):
            cache.set(f'otp_requests_{email}', 0)
            otp_request_count = 0
            cache.set(f'first_otp_request_{email}', timezone.now())

        if otp_request_count >= 6:
            return Response({'error': 'Too many OTP requests. Please try again after 1 hour.'}, status=status.HTTP_429_TOO_MANY_REQUESTS)

        otp = generate_otp()

        user.otp = otp
        user.otp_created_at = timezone.now()
        user.save()

        email_body = f'Hi {user.username}, use the OTP below to reset your password:\n{otp}'
        data = {
            'email_body': email_body,
            'to_email': user.email,
            'email_subject': 'Reset Your Password'
        }

        Util.send_email(data)
        
        cache.set(f'otp_requests_{email}', otp_request_count + 1)
        if not first_request_time:
            cache.set(f'first_otp_request_{email}', timezone.now())

        return Response({'detail': 'OTP sent to your email'}, status=status.HTTP_200_OK)


class SetNewPasswordWithOTPView(generics.GenericAPIView):
    serializer_class = SetNewPasswordWithOTPSerializer

    def patch(self, request):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        serializer.save()

        return Response({'detail': 'Password reset successful'}, status=status.HTTP_200_OK)
