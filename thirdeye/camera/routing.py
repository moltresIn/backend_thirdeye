# camera/routing.py
from django.urls import re_path
from .consumers import CameraConsumer

websocket_urlpatterns = [
    re_path(r'ws/camera/(?P<stream_id>[^/]+)/$', CameraConsumer.as_asgi()),
    #re_path(r'ws/notifications/$', NotificationConsumer.as_asgi()),
]

print('camera/routing.py: WebSocket routing set up for CameraConsumer')
