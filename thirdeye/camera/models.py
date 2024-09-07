#camera/models.py
#from django.db import models
#from django.conf import settings
import os
#from django.utils import timezone
#from urllib.parse import quote
import datetime

# camera/models.py
from django.db import models
from django.conf import settings
from django.utils import timezone
from urllib.parse import quote

class TempFace(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='temp_faces', null=True, blank=True)
    face_id = models.CharField(max_length=100)
    image_data = models.BinaryField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True)  # Store face embedding
    last_seen = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    date_seen = models.DateField(default=timezone.now)  # Store the date of the last seen

    def __str__(self):
        return f"TempFace {self.face_id} (ID: {self.id})"

class SelectedFace(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='selected_faces', null=True, blank=True)
    face_id = models.CharField(max_length=100)
    image_data = models.BinaryField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True)  # Store face embedding
    quality_score = models.FloatField(default=0.0)
    last_seen = models.DateTimeField(default=timezone.now)
    timestamp = models.DateTimeField(default=timezone.now)
    blur_score = models.FloatField(default=0.0)
    is_known = models.BooleanField(default=False)  # Indicates if the face is known
    date_seen = models.DateField(default=timezone.now)  # Store the date of the last seen

    class Meta:
        unique_together = ('user', 'face_id', 'date_seen')

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

class FaceAnalytics(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    total_faces = models.IntegerField(default=0)
    known_faces = models.IntegerField(default=0)
    unknown_faces = models.IntegerField(default=0)
    face_counts = models.JSONField(default=dict)
    timestamp = models.DateTimeField(default=timezone.now)

    known_faces_today = models.IntegerField(default=0)
    known_faces_week = models.IntegerField(default=0)
    known_faces_month = models.IntegerField(default=0)
    known_faces_year = models.IntegerField(default=0)
    known_faces_all = models.IntegerField(default=0)

    class Meta:
        unique_together = ('user', 'date')

    def __str__(self):
        return f"FaceAnalytics for {self.user.username} on {self.date}"
