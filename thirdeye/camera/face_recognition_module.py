# camera/face_recognition.py

import cv2
import numpy as np
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
import os
from datetime import date, datetime
import logging
import asyncio
import torch
from ultralytics import YOLO
from django.utils import timezone
from django.conf import settings
from asgiref.sync import sync_to_async
from django.contrib.auth import get_user_model
from .models import TempFace, SelectedFace

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configuration parameters
MAX_FACES_PER_ID = 15
FACE_SAVE_INTERVAL = 7
PROCESSING_INTERVAL = 1
MAX_COSINE_DISTANCE = 0.3
NN_BUDGET = 300
TRACKER_MAX_AGE = 100

class FaceRecognitionProcessor:
    def __init__(self, user=None):
        self.user = user
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Load YOLO model for face detection
        model_path = os.path.join(settings.BASE_DIR, 'yolov8m-face.pt')
        self.facemodel = YOLO(model_path).to(self.device)
        logger.info(f"YOLO model loaded on device: {self.device}")

        # Initialize DeepSORT tracker
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
        self.tracker = Tracker(metric, max_age=TRACKER_MAX_AGE)
        logger.info("DeepSORT tracker initialized")

        # Initialize counters and mappings
        self.current_date = date.today()
        self.face_id_counter = 1
        self.face_id_mapping = {}
        self.frame_save_counter = {}

        # Initialize frame buffer
        self.frame_buffer = asyncio.Queue(maxsize=10)
        logger.info("FaceRecognitionProcessor initialized")

        # Start periodic processing task
        self.periodic_task = asyncio.create_task(self.periodic_processing())
        logger.info("Periodic processing task started")

    async def process_frame(self, frame):
        logger.debug("Processing new frame")
        await self.frame_buffer.put(frame)
        return await self.process_frame_from_buffer()

    async def process_frame_from_buffer(self):
        frame = await self.frame_buffer.get()
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
                continue

            bbox = track.to_tlbr()
            temp_face = await self.save_face_image(frame, track)

            if temp_face is None:
                continue

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            cv2.putText(frame, temp_face.face_id, (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            detected_faces.append({
                'id': temp_face.id,
                'face_id': temp_face.face_id,
                'last_seen': timezone.localtime(temp_face.last_seen).strftime('%I:%M %p'),
                'image_data': temp_face.image_data,
                'coordinates': {
                    'left': bbox[0],
                    'top': bbox[1],
                    'right': bbox[2],
                    'bottom': bbox[3]
                }
            })
            logger.debug(f"Processed face: {temp_face.face_id}")

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

        return np.array(faces)

    def generate_feature(self, face, frame):
        x, y, w, h, _ = face.astype(int)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return np.zeros(128)

        face_roi = cv2.resize(face_roi, (96, 96))
        return face_roi.flatten() / 255.0

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
            self.face_id_mapping[track_id] = self.get_next_face_id()
            self.frame_save_counter[track_id] = 0

        self.frame_save_counter[track_id] += 1

        if self.frame_save_counter[track_id] % FACE_SAVE_INTERVAL != 0:
            return None

        face_id = self.face_id_mapping[track_id]
        bbox = track.to_tlbr()
        h, w = frame.shape[:2]
        pad_w, pad_h = 0.2 * (bbox[2] - bbox[0]), 0.2 * (bbox[3] - bbox[1])
        x1, y1 = max(0, int(bbox[0] - pad_w)), max(0, int(bbox[1] - pad_h))
        x2, y2 = min(w, int(bbox[2] + pad_w)), min(h, int(bbox[3] + pad_h))

        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            image_count = await sync_to_async(TempFace.objects.filter(face_id=face_id).count)()
            if image_count >= MAX_FACES_PER_ID:
                logger.info(f"Max images reached for {face_id}")
                await self.process_temp_faces()      
                return None

            is_success, buffer = cv2.imencode(".jpg", face_img)
            if not is_success:
                logger.error(f"Failed to encode image for {face_id}")
                return None

            image_data = buffer.tobytes()
            try:
                temp_face = await sync_to_async(self._create_temp_face)(face_id, image_data)
                logger.info(f"Created TempFace for {face_id}")
                return temp_face
            except Exception as e:
                logger.error(f"Error creating TempFace for {face_id}: {str(e)}", exc_info=True)
                return None

        return None

    def _create_temp_face(self, face_id, image_data):
        new_face = TempFace(
            user=self.user,
            face_id=face_id,
            image_data=image_data,
            last_seen=timezone.now()
        )
        new_face.save()
        logger.info(f"Saved TempFace to database: {new_face}")
        return new_face

    async def periodic_processing(self):
        while True:
            try:
                logger.info("Starting periodic processing of temp faces...")
                await self.process_temp_faces()
                logger.info("Finished periodic processing of temp faces")
            except Exception as e:
                logger.error(f"Error in periodic processing: {str(e)}", exc_info=True)
            finally:
                await asyncio.sleep(PROCESSING_INTERVAL)

    async def process_temp_faces(self):
        logger.info("Retrieving unprocessed TempFaces")
        unprocessed_faces = await sync_to_async(list)(
            TempFace.objects.filter(processed=False).order_by('face_id', '-last_seen')
        )
        logger.info(f"Found {len(unprocessed_faces)} unprocessed TempFaces")

        current_face_id = None
        face_group = []

        for face in unprocessed_faces:
            if current_face_id != face.face_id:
                if face_group:
                    await self.process_face_group(face_group)
                current_face_id = face.face_id
                face_group = []

            face_group.append(face)

            if len(face_group) == MAX_FACES_PER_ID:
                await self.process_face_group(face_group)
                face_group = []

        if face_group:
            await self.process_face_group(face_group)

    async def process_face_group(self, face_group):
      if not face_group:
          return

      face_id = face_group[0].face_id
      logger.info(f"Processing face group for face_id: {face_id}")

      best_image, best_quality_score = None, -float('inf')
      last_seen = face_group[0].last_seen

      for face in face_group:
          image_data = await sync_to_async(lambda: face.image_data)()
          if image_data:
              image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

              if image is None:
                  continue

              blur_score = self.detect_blur(image)
              angle_score = self.calculate_face_angle(image)

              quality_score = blur_score - (angle_score / 10)

              if quality_score > best_quality_score:
                  best_quality_score = quality_score
                  best_image = image_data
                  last_seen = face.last_seen

      if best_image is not None:
          logger.info(f"Creating/updating SelectedFace for face_id: {face_id}")
          await self.create_update_selected_face(face_id, best_image, best_quality_score, last_seen)
        
          # Delete TempFace records after processing
          # Delete all TempFace records after successfully creating/updating the SelectedFace
          await sync_to_async(TempFace.objects.all().delete)()
          logger.info("Deleted all TempFace records")
          
      else:
          logger.info(f"No suitable image found for SelectedFace, face_id: {face_id}")

      for face in face_group:
          face.processed = True
          await sync_to_async(face.save)()


    async def create_update_selected_face(self, face_id, image_data, quality_score, last_seen):
      try:
          blur_score = self.detect_blur(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR))
          date_seen = last_seen.date()  # Extract the date part of last_seen
    
          selected_face, created = await sync_to_async(SelectedFace.objects.update_or_create)(
              face_id=face_id,
              user=self.user,
              date_seen=date_seen,  # Ensures uniqueness is per day
              defaults={
                  'image_data': image_data,
                  'quality_score': quality_score,
                  'blur_score': blur_score,
                  'last_seen': last_seen,
                  'timestamp': timezone.now()
              }
          )
          action = "Created" if created else "Updated"
          logger.info(f"{action} SelectedFace for {face_id} on {date_seen}")
      except Exception as e:
          logger.error(f"Error creating/updating SelectedFace for {face_id} on {date_seen}: {str(e)}", exc_info=True)


    def detect_blur(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_face_angle(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            center_x = x + w // 2
            center_y = y + h // 2
            image_height, image_width, _ = image.shape
            angle = np.arctan2(center_y - image_height // 2, center_x - image_width // 2) * 180 / np.pi
            return abs(angle)
        return 180

