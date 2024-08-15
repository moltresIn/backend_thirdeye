#camera/models.py
from django.db import models
from django.conf import settings
import os
from django.utils import timezone
from urllib.parse import quote
import datetime

class TempFace(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    face_id = models.CharField(max_length=20, unique=True)
    image_path = models.CharField(max_length=255, default=list)  # Changed from JSONField to CharField
    last_seen = models.DateTimeField(auto_now=True)

    def save(self, *args, **kwargs):
        if not self.face_id:
            # Generate a unique face_id if not provided
            max_id = TempFace.objects.filter(user=self.user).aggregate(models.Max('face_id'))['face_id__max']
            if max_id:
                num = int(max_id.split('_')[-1]) + 1
            else:
                num = 1
            self.face_id = f"unknown_{num:03d}"
        super().save(*args, **kwargs)


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
