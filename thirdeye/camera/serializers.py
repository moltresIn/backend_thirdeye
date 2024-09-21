#camera/serilizers.py
from rest_framework import serializers
from django.utils import timezone
import base64
from .models import (
    StaticCamera, DDNSCamera, CameraStream, TempFace, 
    SelectedFace, FaceVisit, FaceAnalytics, NotificationLog
)

class TempFaceSerializer(serializers.ModelSerializer):
    last_seen = serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()

    class Meta:
        model = TempFace
        fields = ['id', 'user', 'face_id', 'image', 'last_seen', 'processed']

    def get_last_seen(self, obj):
        """Convert last_seen to local time and format."""
        if obj.last_seen:
            local_time = timezone.localtime(obj.last_seen)
            return local_time.strftime('%I:%M %p')
        return None

    def get_image(self, obj):
        """Encode image data to base64."""
        if obj.image_data:
            return base64.b64encode(obj.image_data).decode('utf-8')
        return None

class FaceVisitSerializer(serializers.ModelSerializer):
    detected_time = serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()

    class Meta:
        model = FaceVisit
        fields = ['detected_time', 'image']

    def get_detected_time(self, obj):
        if obj.detected_time:
            local_time = timezone.localtime(obj.detected_time)
            return local_time.strftime('%I:%M %p')
        return None

    def get_image(self, obj):
        if obj.image_data:
            return base64.b64encode(obj.image_data).decode('utf-8')
        return None

class SelectedFaceSerializer(serializers.ModelSerializer):
    face_visits = serializers.SerializerMethodField()
    total_visits = serializers.SerializerMethodField()

    class Meta:
        model = SelectedFace
        fields = [
            'id', 'user', 'face_id', 'quality_score', 
            'is_known', 'face_visits', 'total_visits'
        ]

    def get_face_visits(self, obj):
        visits = getattr(obj, 'filtered_face_visits', [])
        return FaceVisitSerializer(visits, many=True).data

    def get_total_visits(self, obj):
        return len(getattr(obj, 'filtered_face_visits', []))


class StaticCameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = StaticCamera
        fields = ['ip_address', 'username', 'password', 'name']

class DDNSCameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = DDNSCamera
        fields = ['ddns_hostname', 'username', 'password', 'name']

class CameraStreamSerializer(serializers.ModelSerializer):
    class Meta:
        model = CameraStream
        fields = ['stream_url']

class FaceAnalyticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = FaceAnalytics
        fields = [
            'date', 'total_faces', 'known_faces', 'unknown_faces', 
            'known_faces_today', 'known_faces_week', 'known_faces_month', 
            'known_faces_year', 'facecount'
        ]


class NotificationLogSerializer(serializers.ModelSerializer):
    detected_time = serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()

    class Meta:
        model = NotificationLog
        fields = ['user', 'face_id', 'camera_name', 'detected_time', 'notification_sent', 'image']

    def get_detected_time(self, obj):
        """Convert detected_time to local time and format."""
        if obj.detected_time:
            local_time = timezone.localtime(obj.detected_time)
            return local_time.strftime('%I:%M %p, %Y-%m-%d')
        return None

    def get_image(self, obj):
        """Encode image data to base64."""
        if obj.image_data:
            return base64.b64encode(obj.image_data).decode('utf-8')
        return None
