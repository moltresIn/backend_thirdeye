#camera/ face_recognition_module.py
import cv2
import numpy as np
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
import os
from datetime import date
import logging
import asyncio
import torch
from ultralytics import YOLO
from django.utils import timezone
from django.conf import settings
from django.core.cache import cache
from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from .models import TempFace

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = os.path.join(settings.BASE_DIR, 'yolov8m-face.pt')

User = get_user_model()

class FaceRecognitionProcessor:
    def __init__(self, user=None):
        self.user = user
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.facemodel = YOLO(model_path).to(self.device)
        logger.info(f"YOLO model loaded on device: {self.device}")

        max_cosine_distance = 0.3
        nn_budget = 300
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_age=100)

        self.current_date = date.today()
        self.face_id_counter = 1
        self.face_id_mapping = {}

        self.frame_buffer = asyncio.Queue(maxsize=10)
        self.face_cache = {}

        logger.info("FaceRecognitionProcessor initialized")

    async def process_frame(self, frame):
        await self.frame_buffer.put(frame)
        logger.debug("Frame added to buffer")
        return await self.process_frame_from_buffer()

    async def process_frame_from_buffer(self):
        frame = await self.frame_buffer.get()
        logger.debug("Frame retrieved from buffer for processing")
        
        faces = self.detect_faces(frame)
        logger.debug(f"Detected {len(faces)} faces")
        
        detections = [Detection(face[:4], face[4], self.generate_feature(face, frame)) for face in faces]
        logger.debug(f"Created {len(detections)} detections for tracker")
        
        self.tracker.predict()
        self.tracker.update(detections)
        logger.debug(f"Tracker updated with {len(self.tracker.tracks)} tracks")
        
        detected_faces = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                logger.debug(f"Skipping unconfirmed track {track.track_id}")
                continue
            
            bbox = track.to_tlbr()
            face, image_url = await self.save_face_image(frame, track)
            
            if face is None:
                logger.warning(f"Failed to save face for track {track.track_id}")
                continue

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, face.face_id, (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            detected_faces.append({
                'id': face.id,
                'face_id': face.face_id,
                'last_seen': timezone.localtime(face.last_seen).strftime('%I:%M %p'),
                'image_url': image_url,
                'coordinates': {
                    'left': bbox[0],
                    'top': bbox[1],
                    'right': bbox[2],
                    'bottom': bbox[3]
                }
            })
            logger.debug(f"Processed face: {face.face_id}")
        
        logger.info(f"Processed frame with {len(detected_faces)} faces")
        return frame, detected_faces

    def detect_faces(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.facemodel(frame_rgb, conf=0.5)
        
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                faces.append([x1, y1, x2 - x1, y2 - y1, confidence])
        
        logger.info(f"Detected {len(faces)} faces in frame")
        return np.array(faces)

    def generate_feature(self, face, frame):
        x, y, w, h, _ = face.astype(int)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return np.zeros(128)
        face_roi = cv2.resize(face_roi, (96, 96))
        feature = face_roi.flatten() / 255.0
        return feature

    def get_next_face_id(self):
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.face_id_counter = 1
            self.face_id_mapping.clear()
        
        face_id = f"unknown_{self.face_id_counter:03d}"
        self.face_id_counter += 1
        logger.info(f"Generated new face ID: {face_id}")
        return face_id

    async def save_face_image(self, frame, track):
        track_id = int(track.track_id)
        if track_id not in self.face_id_mapping:
            self.face_id_mapping[track_id] = f"unknown_{self.face_id_counter:03d}"
            self.face_id_counter += 1
        
        face_id = self.face_id_mapping[track_id]

        today = self.current_date.strftime("%Y-%m-%d")
        directory = os.path.join(settings.MEDIA_ROOT, 'faces', today, face_id)
        os.makedirs(directory, exist_ok=True)

        bbox = track.to_tlbr()
        h, w = frame.shape[:2]
        pad_w, pad_h = 0.2 * (bbox[2] - bbox[0]), 0.2 * (bbox[3] - bbox[1])
        x1, y1 = max(0, int(bbox[0] - pad_w)), max(0, int(bbox[1] - pad_h))
        x2, y2 = min(w, int(bbox[2] + pad_w)), min(h, int(bbox[3] + pad_h))

        face_img = frame[y1:y2, x1:x2]

        if face_img.size > 0:
            image_count = len([f for f in os.listdir(directory) if f.endswith('.jpg')])
            filename = os.path.join(directory, f"{face_id}_{image_count:02d}.jpg")
            cv2.imwrite(filename, face_img)

            relative_path = os.path.relpath(filename, settings.MEDIA_ROOT)
            image_url = f"{settings.MEDIA_URL}{relative_path}"

            # Use cache to reduce database queries
            cache_key = f"temp_face_{face_id}"
            temp_face = cache.get(cache_key)
            if not temp_face:
                try:
                    temp_face = await sync_to_async(self._get_or_create_temp_face)(face_id, image_url)
                    cache.set(cache_key, temp_face, timeout=300)  # Cache for 5 minutes
                    logger.info(f"{'Created' if temp_face._state.adding else 'Retrieved'} TempFace for {face_id}")
                except Exception as e:
                    logger.error(f"Error creating/getting TempFace for {face_id}: {str(e)}")
                    return None, None

            temp_face.image_path = image_url
            temp_face.last_seen = timezone.now()
            try:
                await sync_to_async(temp_face.save)()
                logger.info(f"Saved face image for {face_id}")
            except Exception as e:
                logger.error(f"Error saving TempFace for {face_id}: {str(e)}")
                return None, None

            return temp_face, image_url

        return None, None

    def _get_or_create_temp_face(self, face_id, image_url):
        temp_face, created = TempFace.objects.get_or_create(
            face_id=face_id,
            defaults={
                'user': self.user,
                'image_path': image_url,
                'last_seen': timezone.now()
            }
        )
        if created:
            logger.info(f"Created new TempFace for {face_id}")
        else:
            logger.info(f"Retrieved existing TempFace for {face_id}")
        return temp_face
