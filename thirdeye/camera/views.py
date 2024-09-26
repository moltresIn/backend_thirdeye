# camera/views.py
from django.db.models import Prefetch, Exists, OuterRef,Count
from django.utils.timezone import make_aware
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import StaticCamera, DDNSCamera, CameraStream, SelectedFace, TempFace, FaceAnalytics,NotificationLog,FaceVisit
from .serializers import (
    StaticCameraSerializer, DDNSCameraSerializer, CameraStreamSerializer, 
    SelectedFaceSerializer, TempFaceSerializer, FaceAnalyticsSerializer,NotificationLogSerializer
)
from .pagination import DynamicPageSizePagination
from django.db.models import Q
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from django.shortcuts import get_object_or_404
from django.core.cache import cache
from django.utils.dateparse import parse_date
from django.utils import timezone
from datetime import datetime, timedelta,time
import pytz
import logging
from .face_recognition_module import FaceRecognitionProcessor
from asgiref.sync import async_to_sync
from django.db import transaction
from .permissions import IsPaidUser, IsTrialUser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from authentication.models import Subscription, UserRole
from django.utils import timezone

logger = logging.getLogger(__name__)

class StaticCameraView(generics.GenericAPIView):
    serializer_class = StaticCameraSerializer
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(request_body=StaticCameraSerializer)
    def post(self, request):
        logger.info('StaticCameraView.post: Received POST request')
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            logger.info('StaticCameraView.post: Serializer is valid')
            static_camera = StaticCamera.objects.create(user=request.user, **serializer.validated_data)
            logger.info(f'StaticCameraView.post: Created StaticCamera: {static_camera}')
            CameraStream.objects.create(user=request.user, camera=static_camera, stream_url=static_camera.rtsp_url())
            logger.info('StaticCameraView.post: Created CameraStream for StaticCamera')
            return Response({"message": "Static camera details saved successfully"}, status=status.HTTP_201_CREATED)
        logger.error(f'StaticCameraView.post: Serializer errors: {serializer.errors}')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DDNSCameraView(generics.GenericAPIView):
    serializer_class = DDNSCameraSerializer
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(request_body=DDNSCameraSerializer)
    def post(self, request):
        logger.info('DDNSCameraView.post: Received POST request')
        serializer = self.serializer_class(data=request.data)
        if serializer.is_valid():
            logger.info('DDNSCameraView.post: Serializer is valid')
            ddns_camera = DDNSCamera.objects.create(user=request.user, **serializer.validated_data)
            logger.info(f'DDNSCameraView.post: Created DDNSCamera: {ddns_camera}')
            CameraStream.objects.create(user=request.user, ddns_camera=ddns_camera, stream_url=ddns_camera.rtsp_url())
            logger.info('DDNSCameraView.post: Created CameraStream for DDNSCamera')
            return Response({"message": "DDNS camera details saved successfully"}, status=status.HTTP_201_CREATED)
        logger.error(f'DDNSCameraView.post: Serializer errors: {serializer.errors}')
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GetStreamURLView(APIView):
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        manual_parameters=[
            openapi.Parameter('camera_type', openapi.IN_PATH, description="Type of camera (static or ddns)", type=openapi.TYPE_STRING),
        ],
        responses={200: openapi.Response('Stream URLs', CameraStreamSerializer(many=True))},
    )
    def get(self, request, camera_type):
        logger.info(f'GetStreamURLView.get: Received GET request for {camera_type} cameras')
        try:
            cache_key = f"stream_urls_{camera_type}_{request.user.id}"
            stream_urls = cache.get(cache_key)
            
            if stream_urls is None:
                if camera_type == 'static':
                    cameras = StaticCamera.objects.filter(user=request.user)
                elif camera_type == 'ddns':
                    cameras = DDNSCamera.objects.filter(user=request.user)
                else:
                    logger.error('GetStreamURLView.get: Invalid camera type')
                    return Response({"error": "Invalid camera type"}, status=status.HTTP_400_BAD_REQUEST)

                if not cameras.exists():
                    logger.warning(f'GetStreamURLView.get: No {camera_type} cameras found')
                    return Response({"error": f"No {camera_type} cameras found"}, status=status.HTTP_404_NOT_FOUND)

                stream_urls = []
                for camera in cameras:
                    streams = CameraStream.objects.filter(
                        Q(user=request.user),
                        Q(camera=camera) if camera_type == 'static' else Q(ddns_camera=camera)
                    )
                    for stream in streams:
                        ws_url = f"ws://{request.get_host()}/ws/camera/{stream.id}/"
                        stream_urls.append({
                            "id": stream.id,
                            "name": camera.name,
                            "url": ws_url
                        })
                
                cache.set(cache_key, stream_urls, 300)

            logger.info(f'GetStreamURLView.get: Returning {len(stream_urls)} stream URLs')
            return Response({"stream_urls": stream_urls}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f'GetStreamURLView.get: Exception occurred: {str(e)}')
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





class FaceView(generics.ListAPIView):
    serializer_class = SelectedFaceSerializer
    permission_classes = [IsAuthenticated, IsPaidUser]
    pagination_class = DynamicPageSizePagination

    def get_queryset(self):
        logger.info('FaceView.get_queryset: Building queryset for SelectedFace')
        date_str = self.request.query_params.get('date')
        is_known = self.request.query_params.get('is_known')

        queryset = SelectedFace.objects.filter(user=self.request.user)

        if date_str:
            try:
                date = timezone.datetime.strptime(date_str, '%Y-%m-%d').date()
                queryset = queryset.filter(face_visits__date_seen=date).distinct()
                
                # Prefetch related FaceVisit objects for the specific date
                queryset = queryset.prefetch_related(
                    Prefetch('face_visits', 
                             queryset=FaceVisit.objects.filter(date_seen=date).order_by('-detected_time'),
                             to_attr='filtered_face_visits')
                )
            except ValueError:
                logger.error(f'FaceView.get_queryset: Invalid date format: {date_str}')
                return queryset.none()
        else:
            # If no date is specified, don't prefetch any FaceVisit objects
            queryset = queryset.none()

        if is_known is not None:
            is_known = is_known.lower() == 'true'
            queryset = queryset.filter(is_known=is_known)

        queryset = queryset.order_by('-last_seen')

        logger.info(f'FaceView.get_queryset: Returning queryset with {queryset.count()} items')
        return queryset

    def get_serializer_context(self):
        context = super().get_serializer_context()
        context['date_seen'] = self.request.query_params.get('date')
        return context




class RenameFaceView(APIView):
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'old_face_id': openapi.Schema(type=openapi.TYPE_STRING),
                'new_face_id': openapi.Schema(type=openapi.TYPE_STRING),
            },
            required=['old_face_id', 'new_face_id']
        ),
        responses={200: openapi.Response('Face renamed successfully')}
    )
    def post(self, request):
        old_face_id = request.data.get('old_face_id')
        new_face_id = request.data.get('new_face_id')

        if not old_face_id or not new_face_id:
            return Response({"error": "Both old_face_id and new_face_id are required"}, status=400)

        # Initialize the processor
        face_processor = FaceRecognitionProcessor(user=request.user)

        # Since rename_face is an async method, we need to convert it to sync using async_to_sync
        async_to_sync(face_processor.rename_face)(old_face_id, new_face_id)

        return Response({"message": "Face renamed successfully"}, status=200)



class FaceAnalyticsView(APIView):
    permission_classes = [IsAuthenticated, IsPaidUser]

    def get(self, request):
        try:
            # Initialize the face analytics processor with the authenticated user
            face_processor = FaceRecognitionProcessor(user=request.user)

            # Get the analytics data
            analytics = face_processor.get_face_analytics()

            if analytics:
                with transaction.atomic():
                    try:
                        # Try to get existing entry for today
                        face_analytics = FaceAnalytics.objects.get(
                            user=request.user,
                            date=analytics['date']
                        )
                        # Update existing entry
                        for key, value in analytics.items():
                            setattr(face_analytics, key, value)
                        face_analytics.save()
                    except FaceAnalytics.DoesNotExist:
                        # Create a new entry if it doesn't exist
                        face_analytics = FaceAnalytics.objects.create(
                            user=request.user,
                            date=analytics['date'],
                            total_faces=analytics['total_faces'],
                            known_faces=analytics['known_faces'],
                            unknown_faces=analytics['unknown_faces'],
                            known_faces_today=analytics['known_faces_today'],
                            known_faces_week=analytics['known_faces_week'],
                            known_faces_month=analytics['known_faces_month'],
                            known_faces_year=analytics['known_faces_year'],
                            facecount=analytics['facecount']
                        )

                # Serialize the response and return the data
                serializer = FaceAnalyticsSerializer(face_analytics)
                return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                return Response({"error": "Failed to retrieve face analytics"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        except Exception as e:
            logger.error(f"Error in FaceAnalyticsView: {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)








class RenameCameraView(APIView):
    permission_classes = [IsAuthenticated]
    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            required=['name'],
            properties={
                'name': openapi.Schema(type=openapi.TYPE_STRING, description='New name of the camera'),
            },
        ),
        responses={200: 'OK', 400: 'Bad Request', 404: 'Not Found'},
    )
    def patch(self, request, camera_type, pk):
        logger.info(f'RenameCameraView.patch: Received PATCH request for {camera_type} camera with pk {pk}')
        name = request.data.get('name')
        if not name:
            logger.error('RenameCameraView.patch: Missing name in request data')
            return Response({"name": ["This field is required."]}, status=status.HTTP_400_BAD_REQUEST)

        try:
            if camera_type == 'static':
                camera = StaticCamera.objects.get(pk=pk, user=request.user)
            elif camera_type == 'ddns':
                camera = DDNSCamera.objects.get(pk=pk, user=request.user)
            else:
                logger.error('RenameCameraView.patch: Invalid camera type')
                return Response({"error": "Invalid camera type"}, status=status.HTTP_400_BAD_REQUEST)
        except (StaticCamera.DoesNotExist, DDNSCamera.DoesNotExist):
            logger.error(f'RenameCameraView.patch: {camera_type.capitalize()} Camera not found')
            return Response({"error": f"{camera_type.capitalize()} Camera not found"}, status=status.HTTP_404_NOT_FOUND)

        camera.name = name
        camera.save()
        logger.info(f'RenameCameraView.patch: Camera {pk} renamed to {name}')

        cache.delete(f"stream_urls_{camera_type}_{request.user.id}")

        return Response({"message": "Camera renamed successfully"}, status=status.HTTP_200_OK)




class NotificationLogView(generics.ListAPIView):
    queryset = NotificationLog.objects.all()
    serializer_class = NotificationLogSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return NotificationLog.objects.filter(user=self.request.user).order_by('-detected_time')


class ActivateSubscriptionView(APIView):
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'days': openapi.Schema(
                    type=openapi.TYPE_INTEGER, 
                    description='Number of days for subscription activation (default is 30)',
                    default=30
                ),
            },
            required=[]
        ),
        responses={200: openapi.Response('Subscription activated successfully')}
    )
    def post(self, request):
        days = request.data.get('days', 30)  # Default to 30 days if not specified
        subscription, _ = Subscription.objects.get_or_create(user=request.user)
        subscription.is_trial = False
        subscription.is_active = True
        subscription.start_date = timezone.now()
        subscription.end_date = timezone.now() + timezone.timedelta(days=days)
        subscription.save()

        user_role, _ = UserRole.objects.get_or_create(user=request.user)
        user_role.role = UserRole.PAID
        user_role.save()

        return Response({"message": "Subscription activated successfully"}, status=status.HTTP_200_OK)
