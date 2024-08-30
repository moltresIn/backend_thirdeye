
# camera/serializers.py

from rest_framework import serializers
from .models import StaticCamera, DDNSCamera, CameraStream, TempFace,SelectedFace
from django.utils import timezone
import base64

#from rest_framework import serializers
#from .models import TempFace, SelectedFace
#import base64
#from django.utils import timezone

class TempFaceSerializer(serializers.ModelSerializer):
    last_seen = serializers.SerializerMethodField()
    image = serializers.SerializerMethodField()

    class Meta:
        model = TempFace
        fields = ['id', 'user', 'face_id', 'image', 'last_seen']

    def get_last_seen(self, obj):
        if obj.last_seen:
            local_time = timezone.localtime(obj.last_seen)
            return local_time.strftime('%I:%M %p')
        return None

    def get_image(self, obj):
        if obj.image_data:
            return base64.b64encode(obj.image_data).decode('utf-8')
        return None

class SelectedFaceSerializer(serializers.ModelSerializer):
    image = serializers.SerializerMethodField()
    last_seen = serializers.SerializerMethodField()

    class Meta:
        model = SelectedFace
        fields = ['id', 'user', 'face_id', 'image', 'quality_score', 'last_seen', 'timestamp','date_seen']

    def get_image(self, obj):
        if obj.image_data:
            return base64.b64encode(obj.image_data).decode('utf-8')
        return None

    def get_last_seen(self, obj):
        if obj.last_seen:
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
