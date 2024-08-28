# camera/views.py
from rest_framework import status, generics
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .models import StaticCamera, DDNSCamera, CameraStream, SelectedFace,TempFace
from .serializers import (
    StaticCameraSerializer, DDNSCameraSerializer, CameraStreamSerializer, 
    SelectedFaceSerializer, TempFaceSerializer
)
from .pagination import DynamicPageSizePagination
from django.db.models import Q
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.shortcuts import get_object_or_404
from django.core.cache import cache
from django.utils import timezone
from datetime import datetime, timedelta
import pytz
import logging

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
#class SelectedFaceView(generics.ListAPIView):
    serializer_class = SelectedFaceSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = DynamicPageSizePagination

    def get_queryset(self):
        logger.info('SelectedFaceView.get_queryset: Building queryset for SelectedFace')
        queryset = SelectedFace.objects.filter(user=self.request.user).order_by('-last_seen')
        
        logger.info(f'Initial queryset count: {queryset.count()}')
        logger.info(f'User: {self.request.user}')

        # Log all SelectedFace objects for this user
        all_faces = list(queryset.values('id', 'last_seen'))
        logger.info(f'All SelectedFace objects for user: {all_faces}')

        filters_applied = []

        date_str = self.request.query_params.get('date')

        if date_str:
            try:
                # Parse the date string
                local_tz = pytz.timezone('Asia/Kolkata')
                date = datetime.strptime(date_str, '%Y-%m-%d').date()
                
                # Create a datetime range for the entire day in local time
                start_datetime = local_tz.localize(datetime.combine(date, datetime.min.time()))
                end_datetime = start_datetime + timedelta(days=1)
                
                # Convert to UTC for database query
                start_datetime_utc = start_datetime.astimezone(pytz.UTC)
                end_datetime_utc = end_datetime.astimezone(pytz.UTC)
                
                logger.info(f'Filtering by date range (local): {start_datetime} to {end_datetime}')
                logger.info(f'Filtering by date range (UTC): {start_datetime_utc} to {end_datetime_utc}')
                
                # Apply the filter
                queryset = queryset.filter(last_seen__gte=start_datetime_utc, last_seen__lt=end_datetime_utc)
                filters_applied.append(f'date={date}')
                
                logger.info(f'Queryset count after date filter: {queryset.count()}')
                
                # Log the SQL query
                logger.info(f'SQL query: {queryset.query}')
                
                # Log all SelectedFace objects after filter
                filtered_faces = list(queryset.values('id', 'last_seen'))
                logger.info(f'SelectedFace objects after filter: {filtered_faces}')
                
            except ValueError:
                logger.error(f'SelectedFaceView.get_queryset: Invalid date format: {date_str}')
                return queryset.none()

        logger.info(f'SelectedFaceView.get_queryset: Applied filters: {", ".join(filters_applied)}')
        logger.info(f'SelectedFaceView.get_queryset: Returning queryset with {queryset.count()} items')
        
        return queryset

    def list(self, request, *args, **kwargs):
        logger.info('SelectedFaceView.list: Received list request')
        try:
            queryset = self.get_queryset()
        
            logger.info(f'Queryset count before pagination: {queryset.count()}')
        
            page = self.paginate_queryset(queryset)
        
            if page is not None:
                logger.info(f'Page count: {len(page)}')
                serializer = self.get_serializer(page, many=True, context={'request': request})
                response = self.get_paginated_response(serializer.data)
            
                # Log pagination details
                current_page = self.paginator.page.number
                page_size = self.paginator.page_size
                total_pages = self.paginator.page.paginator.num_pages
                logger.info(f'SelectedFaceView.list: Returning page {current_page} of {total_pages} (page size: {page_size})')
            
                return response
        
            logger.warning('SelectedFaceView.list: Pagination not applied, returning all data')
            serializer = self.get_serializer(queryset, many=True, context={'request': request})
            return Response(serializer.data)
        except Exception as e:
            logger.error(f'Error in SelectedFaceView.list: {str(e)}', exc_info=True)
            raise
    


class RenameFaceView(generics.UpdateAPIView):
    queryset = SelectedFace.objects.all()
    serializer_class = SelectedFaceSerializer
    permission_classes = [IsAuthenticated]

    @swagger_auto_schema(
        request_body=SelectedFaceSerializer,
        responses={200: SelectedFaceSerializer()},
    )
    def get_object(self):
        logger.info(f'RenameFaceView.get_object: Getting SelectedFace object with pk: {self.kwargs["pk"]}')
        obj = get_object_or_404(SelectedFace, pk=self.kwargs['pk'], user=self.request.user)
        logger.info(f'RenameFaceView.get_object: Found SelectedFace object: {obj}')
        return obj

    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)
        logger.info(f'RenameFaceView.update: Updated SelectedFace {instance.id} with new name: {instance.face_id}')
        return Response(serializer.data)

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
