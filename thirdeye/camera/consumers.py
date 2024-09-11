# camera/consumers.py

import json
import asyncio
import cv2
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import CameraStream
from .face_recognition_module import FaceRecognitionProcessor
import logging
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from rest_framework_simplejwt.tokens import AccessToken

logger = logging.getLogger(__name__)
User = get_user_model()

class CameraConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        logger.info('WebSocket connection requested')
        self.stream_id = self.scope['url_route']['kwargs']['stream_id']

        # Extract token and get user
        token = self.scope['query_string'].decode().split('=')[-1]
        self.user = await self.get_user_from_token(token)

        if self.user is None or isinstance(self.user, AnonymousUser):
            logger.error('User is not authenticated')
            await self.close()
            return

        # Retrieve camera stream asynchronously using sync_to_async
        self.camera_stream = await sync_to_async(CameraStream.objects.get)(id=self.stream_id)

        # Retrieve the camera name in an async-safe way
        self.camera_name = await self.get_camera_name(self.camera_stream)

        logger.info(f"User {self.user} connected to camera: {self.camera_name}")

        # Initialize face recognition processor
        self.face_processor = FaceRecognitionProcessor(user=self.user, camera_name=self.camera_name)

        # Join notification group
        self.notification_group_name = f"notifications_{self.user.id}"
        await self.channel_layer.group_add(self.notification_group_name, self.channel_name)

        self.stop_stream = False
        self.frame_count = 0
        self.last_frame_time = 0

        # Start periodic face processing task
        self.periodic_task = asyncio.create_task(self.face_processor.periodic_processing())

        await self.accept()
        logger.info(f'WebSocket connection established for stream ID: {self.stream_id}')

        # Start streaming video from the camera
        self.stream_task = asyncio.create_task(self.start_stream())

    async def get_camera_name(self, camera_stream):
        """
        Helper method to asynchronously fetch the camera name.
        This method checks both 'camera' and 'ddns_camera' fields.
        """
        if await sync_to_async(lambda: hasattr(camera_stream, 'camera'))():
            return await sync_to_async(lambda: camera_stream.camera.name)()
        elif await sync_to_async(lambda: hasattr(camera_stream, 'ddns_camera'))():
            return await sync_to_async(lambda: camera_stream.ddns_camera.name)()
        else:
            return "Unknown Camera"

    async def disconnect(self, close_code):
        logger.info(f'WebSocket disconnected with code {close_code}')
        self.stop_stream = True

        # Cancel tasks and clean up
        if hasattr(self, 'stream_task'):
            self.stream_task.cancel()
        if hasattr(self, 'periodic_task'):
            self.periodic_task.cancel()

        await self.cleanup()

        # Remove user from notification group
        await self.channel_layer.group_discard(self.notification_group_name, self.channel_name)

    async def receive(self, text_data):
        logger.info(f'Received data: {text_data}')
        data = json.loads(text_data)

        if data.get('command') == 'stop_stream':
            logger.info('Stop stream command received')
            self.stop_stream = True

    async def start_stream(self):
        max_retries = 10
        frame_skip = 2  # Process every 2nd frame

        for attempt in range(max_retries):
            try:
                stream = await sync_to_async(CameraStream.objects.get)(id=self.stream_id)
                logger.info(f'Retrieved stream: {stream}')

                cap = cv2.VideoCapture(stream.stream_url)
                logger.info(f'Opened video capture for URL: {stream.stream_url}')

                if not cap.isOpened():
                    raise Exception("Failed to open video capture")

                while not self.stop_stream:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning('Failed to capture frame')
                        await asyncio.sleep(0.1)
                        continue

                    self.frame_count += 1
                    if self.frame_count % frame_skip != 0:
                        continue

                    processed_frame, detected_faces = await self.face_processor.process_frame(frame)

                    # Encode frame as base64
                    _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    base64_frame = base64.b64encode(buffer).decode('utf-8')

                    for face in detected_faces:
                        if 'image_data' in face and face['image_data'] is not None:
                            face['image_data'] = base64.b64encode(face['image_data']).decode('utf-8')

                    await self.send(text_data=json.dumps({
                        'frame': base64_frame,
                        'detected_faces': detected_faces,
                    }))

                    if self.frame_count % 30 == 0:
                        current_time = asyncio.get_event_loop().time()
                        fps = 30 / (current_time - self.last_frame_time)
                        logger.info(f'Current FPS: {fps:.2f}')
                        self.last_frame_time = current_time

                    await asyncio.sleep(0.033)

                logger.info('Stream stopped normally')
                break

            except asyncio.CancelledError:
                logger.info('Stream task was cancelled')
                break
            except Exception as e:
                logger.error(f'Error in start_stream (attempt {attempt + 1}/{max_retries}): {str(e)}', exc_info=True)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    await self.send(text_data=json.dumps({'error': str(e)}))
            finally:
                if 'cap' in locals():
                    cap.release()
                    logger.info('Video capture released')

        logger.info('Closing WebSocket connection')
        await self.close()

    async def send_notification(self, event):
        logger.info(f"Sending notification to user {self.user}: {event['message']}")
        await self.send(text_data=json.dumps({'notification': event['message']}))

    async def get_user_from_token(self, token):
        try:
            access_token = AccessToken(token)
            user = await sync_to_async(User.objects.get)(id=access_token['user_id'])
            return user
        except Exception as e:
            logger.error(f'Error authenticating user with token: {str(e)}', exc_info=True)
            return AnonymousUser()

    async def cleanup(self):
        logger.info("Performing cleanup operations")
