#camera/serializers.py
from rest_framework import serializers
from .models import StaticCamera, DDNSCamera, CameraStream, TempFace
from django.utils import timezone

class TempFaceSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    last_seen = serializers.SerializerMethodField()

    class Meta:
        model = TempFace
        fields = ['id', 'user', 'face_id', 'image_path', 'last_seen', 'image_url']

    def get_image_url(self, obj):
        request = self.context.get('request')
        if obj.image_path and request is not None:
            return request.build_absolute_uri(obj.image_path)
        return None

    def get_last_seen(self, obj):
        if obj.last_seen:
            # Convert to local time and format
            local_time = timezone.localtime(obj.last_seen)
            return local_time.strftime('%I:%M %p')
        return None

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
