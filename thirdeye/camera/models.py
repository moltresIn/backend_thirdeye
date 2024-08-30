#camera/models.py
from django.db import models
from django.conf import settings
import os
from django.utils import timezone
from urllib.parse import quote
import datetime

# camera/models.py

#from django.db import models
#from django.contrib.auth import get_user_model
#from django.utils import timezone

#User = get_user_model()

class TempFace(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE , related_name='temp_faces',null=True, blank=True)
    face_id = models.CharField(max_length=100)
    image_data = models.BinaryField( null=True, blank=True)
    last_seen = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)

    def __str__(self):
        return f"TempFace {self.face_id} (ID: {self.id})"

class SelectedFace(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='selected_faces',null=True, blank=True)
    face_id = models.CharField(max_length=100)
    image_data = models.BinaryField( null=True, blank=True)
    quality_score = models.FloatField(default=0.0)
    last_seen = models.DateTimeField(default=timezone.now)
    timestamp = models.DateTimeField(default=timezone.now)
    blur_score = models.FloatField(default=0.0)  
    date_seen = models.DateField(default=timezone.now)  # New field to store the date

    class Meta:
        unique_together = ('user', 'face_id')

    def __str__(self):
        return f"SelectedFace {self.face_id} (ID: {self.id})"


class StaticCamera(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    ip_address = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    name = models.CharField(max_length=255, default="Static Camera")

    def rtsp_url(self):
        encoded_username = quote(self.username)
        encoded_password = quote(self.password)
        return f"rtsp://{encoded_username}:{encoded_password}@{self.ip_address}:1024/Streaming/Channels/101"
class DDNSCamera(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    ddns_hostname = models.CharField(max_length=255)
    username = models.CharField(max_length=255)
    password = models.CharField(max_length=255)
    name = models.CharField(max_length=255, default="DDNS Camera")

    def rtsp_url(self):
        encoded_username = quote(self.username)
        encoded_password = quote(self.password)
        return f"rtsp://{encoded_username}:{encoded_password}@{self.ddns_hostname}:554/Streaming/Channels/101"

class CameraStream(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    camera = models.ForeignKey(StaticCamera, null=True, blank=True, on_delete=models.CASCADE)
    ddns_camera = models.ForeignKey(DDNSCamera, null=True, blank=True, on_delete=models.CASCADE)
    stream_url = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)
