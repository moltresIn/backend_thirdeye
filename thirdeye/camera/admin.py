# camera/admin.py
from django.contrib import admin
from .models import StaticCamera, DDNSCamera, TempFace,  CameraStream

@admin.register(StaticCamera)
class StaticCameraAdmin(admin.ModelAdmin):
    list_display = ('user', 'ip_address', 'name')

@admin.register(DDNSCamera)
class DDNSCameraAdmin(admin.ModelAdmin):
    list_display = ('user', 'ddns_hostname', 'name')

@admin.register(TempFace)
class TempFaceAdmin(admin.ModelAdmin):
    list_display = ('user', 'face_id','last_seen','image_data')



@admin.register(CameraStream)
class CameraStreamAdmin(admin.ModelAdmin):
    list_display = ('user', 'stream_url', 'created_at')
