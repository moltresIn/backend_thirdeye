#camera/models.py
from django.db import models
from django.conf import settings
from django.utils import timezone
from urllib.parse import quote
import datetime

# Model to temporarily store face data before processing
class TempFace(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='temp_faces', null=True, blank=True)
    face_id = models.CharField(max_length=100)
    image_data = models.BinaryField(null=True, blank=True)
    embedding = models.JSONField(null=True, blank=True)  # Store face embedding
    last_seen = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)
    date_seen = models.DateField(default=timezone.now)  # Store the date of the last seen
    quality_score = models.FloatField(default=0.0)

    def __str__(self):
        return f"TempFace {self.face_id} (ID: {self.id})"


# Model to store the processed, identified face
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


# Model to store each instance a face is detected
class FaceVisit(models.Model):
    selected_face = models.ForeignKey(SelectedFace, on_delete=models.CASCADE, related_name='face_visits')
    image_data = models.BinaryField(null=True, blank=True)
    detected_time = models.DateTimeField(default=timezone.now)
    date_seen = models.DateField(default=timezone.now)  # Add this line

    def __str__(self):
        return f"FaceVisit for {self.selected_face.face_id} at {self.detected_time}"

    def save(self, *args, **kwargs):
        if not self.date_seen:
            self.date_seen = self.detected_time.date()
        super().save(*args, **kwargs)

# Model to store static camera details
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

    def __str__(self):
        return f"StaticCamera {self.name} (IP: {self.ip_address})"


# Model to store DDNS camera details
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

    def __str__(self):
        return f"DDNSCamera {self.name} (Hostname: {self.ddns_hostname})"


# Model to store the camera stream URL
class CameraStream(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    camera = models.ForeignKey(StaticCamera, null=True, blank=True, on_delete=models.CASCADE)
    ddns_camera = models.ForeignKey(DDNSCamera, null=True, blank=True, on_delete=models.CASCADE)
    stream_url = models.CharField(max_length=255)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"CameraStream {self.stream_url}"


class FaceAnalytics(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)  # The date for the analytics

    # Total faces (includes both known and unknown)
    total_faces = models.IntegerField(default=0)  # Total faces detected from the start until today
    known_faces = models.IntegerField(default=0)  # Total known faces from the start until today
    unknown_faces = models.IntegerField(default=0)  # Total unknown faces from the start until today
    
    # Period-based known faces
    known_faces_today = models.IntegerField(default=0)
    known_faces_week = models.IntegerField(default=0)
    known_faces_month = models.IntegerField(default=0)
    known_faces_year = models.IntegerField(default=0)

    # `facecount` for known persons (cumulative count)
    facecount = models.JSONField(default=dict)  # Example format: { "face_id": "count" }

    timestamp = models.DateTimeField(default=timezone.now)  # When the analytics were last updated

    class Meta:
        unique_together = ('user', 'date')  # Each user has unique analytics for each day

    def __str__(self):
        return f"FaceAnalytics for {self.user.username} on {self.date}"

# Model for logging notifications sent to users when a face is detected
class NotificationLog(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    face_id = models.CharField(max_length=100)
    camera_name = models.CharField(max_length=255)
    detected_time = models.DateTimeField(default=timezone.now)
    notification_sent = models.BooleanField(default=False)
    image_data = models.BinaryField(null=True, blank=True)

    def __str__(self):
        return f"NotificationLog for {self.face_id} (Sent: {self.notification_sent})"
