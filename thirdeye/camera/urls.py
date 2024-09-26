#camera/urls.py
from django.urls import path
from .views import StaticCameraView, DDNSCameraView, GetStreamURLView, FaceView, RenameFaceView, RenameCameraView,FaceAnalyticsView,NotificationLogView,ActivateSubscriptionView


urlpatterns = [
    path('static-camera/', StaticCameraView.as_view(), name='static_camera'),
    path('ddns-camera/', DDNSCameraView.as_view(), name='ddns_camera'),
    path('get-stream-url/<str:camera_type>/', GetStreamURLView.as_view(), name='get_stream_url'),
    path('faces/', FaceView.as_view(), name='face'),
    path('rename-face/', RenameFaceView.as_view(), name='rename_face'),
    path('rename-camera/<str:camera_type>/<int:pk>/', RenameCameraView.as_view(), name='rename_camera'),
    path('face-analytics/', FaceAnalyticsView.as_view(), name='face_analytics'),  # New path for FaceAnalyticsView
    path('notifications/', NotificationLogView.as_view(), name='notification-log'),
    path('activate-subscription/', ActivateSubscriptionView.as_view(), name='activate-subscription'),


]
