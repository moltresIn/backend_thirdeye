# camera/consumers.py

import json
import asyncio
import cv2
import base64
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import CameraStream
from .face_recognition_module import FaceRecognitionProcessor
import logging
from django.core.cache import cache
from channels.exceptions import StopConsumer
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.conf import settings
from rest_framework_simplejwt.tokens import AccessToken

logger = logging.getLogger(__name__)

User = get_user_model()

class CameraConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        logger.info('WebSocket connection requested')
        self.stream_id = self.scope['url_route']['kwargs']['stream_id']
        
        token = self.scope['query_string'].decode().split('=')[-1]
        logger.debug(f"Extracted token: {token}")

        self.user = await self.get_user_from_token(token)

        if self.user is None or isinstance(self.user, AnonymousUser):
            logger.error('User is not authenticated')
            await self.close()
            return
        
        self.stop_stream = False
        self.frame_count = 0
        self.last_frame_time = 0
        
        self.face_processor = FaceRecognitionProcessor(user=self.user)
        self.periodic_task = asyncio.create_task(self.face_processor.periodic_processing())
        
        await self.accept()
        logger.info(f'WebSocket connection established for stream ID: {self.stream_id}')
        
        self.stream_task = asyncio.create_task(self.start_stream())

    async def disconnect(self, close_code):
        logger.info(f'WebSocket disconnected with code {close_code}')
        self.stop_stream = True
        if hasattr(self, 'stream_task'):
            self.stream_task.cancel()
        if hasattr(self, 'periodic_task'):
            self.periodic_task.cancel()
        
        await self.cleanup()

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
