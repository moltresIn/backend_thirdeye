#camera/notifications.py


from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import json

def send_face_detection_notification(user_id, data):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"notifications_{user_id}",
        {
            "type": "send_notification",
            "message": json.dumps(data)
        }
    )
