# camera/face_recognition.py
from django.db.models import Sum
from django.db import transaction
import base64
import cv2
import numpy as np
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
import os
from datetime import date, timedelta,datetime
import logging
import asyncio
import torch
from ultralytics import YOLO
from django.utils import timezone
from django.conf import settings
from asgiref.sync import sync_to_async
from .models import TempFace, SelectedFace, NotificationLog,FaceVisit,FaceAnalytics
from .serializers import FaceAnalyticsSerializer
import face_recognition
from django.db.models import Count, Q
from channels.layers import get_channel_layer
import pytz 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

IST = pytz.timezone('Asia/Kolkata')  # Define Indian Standard Time

# Configuration parameters
MAX_FACES_PER_ID = 15
FACE_SAVE_INTERVAL = 7
PROCESSING_INTERVAL = 1
MAX_COSINE_DISTANCE = 0.3
NN_BUDGET = 300
TRACKER_MAX_AGE = 100
FACE_MATCH_THRESHOLD = 0.6

class FaceRecognitionProcessor:
    def __init__(self, user=None, camera_name=None):
        self.user = user
        self.camera_name = camera_name
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

        self.face_match_threshold = FACE_MATCH_THRESHOLD
        self.current_date = date.today()
        self.face_id_counter = 1
        self.face_id_mapping = {}
        self.frame_save_counter = {}
        self.available_face_ids = []
        self.frame_buffer = asyncio.Queue(maxsize=10)
        self.in_frame_tracker = {}  # Track if a face is currently in the frame
        logger.info("FaceRecognitionProcessor initialized")

        # Initialize face encoder
        self.face_encoder = face_recognition.face_encodings
        logger.info("Face encoder initialized")

    async def start_periodic_task(self):
        self.periodic_task = asyncio.create_task(self.periodic_processing())
        logger.info("Periodic processing task started")

    async def process_frame(self, frame):
        logger.debug("Processing new frame")
        await self.frame_buffer.put(frame)
        return await self.process_frame_from_buffer()

    async def process_frame_from_buffer(self):
      frame = await self.frame_buffer.get()

      # Step 1: Detect multiple faces in the frame
      faces = self.detect_faces(frame)
      logger.debug(f"Detected {len(faces)} faces in the frame")

      # Step 2: Create detection objects for each detected face
      detections = [Detection(face[:4], face[4], self.generate_feature(face, frame)) for face in faces]
      logger.debug(f"Created {len(detections)} detections for the tracker")

      # Step 3: Use the tracker to update face positions
      self.tracker.predict()
      self.tracker.update(detections)
      logger.debug(f"Tracker updated with {len(self.tracker.tracks)} tracks")

      detected_faces = []
      for track in self.tracker.tracks:
          if not track.is_confirmed() or track.time_since_update > 1:
              continue

          bbox = track.to_tlbr()
          track_id = track.track_id

          # Skip processing if the face is still in the frame and already processed
          if track_id in self.in_frame_tracker:
              logger.debug(f"Skipping already processed face with track_id: {track_id}")
              continue

          # Process and store this face as it's entering the frame
          temp_face = await self.save_face_image(frame, track)
          if temp_face is None:
              continue
 
          self.in_frame_tracker[track_id] = True  # Mark the face as processed
 
          last_seen_ist = temp_face.last_seen.astimezone(IST)
          formatted_last_seen = last_seen_ist.strftime('%I:%M %p')
 
          detected_faces.append({
              'id': temp_face.id,
              'face_id': temp_face.face_id,
              'last_seen': formatted_last_seen,
              'image_data': temp_face.image_data,
              'coordinates': {
                  'left': bbox[0],
                  'top': bbox[1],
                  'right': bbox[2],
                  'bottom': bbox[3]
              }
          })
          logger.debug(f"Stored new face: {temp_face.face_id}")

      # Remove faces that have left the frame
      self.cleanup_exited_faces()
 
      return frame, detected_faces


    def cleanup_exited_faces(self):
      # Remove tracks for faces that have left the frame
      for track in self.tracker.tracks:
          if track.time_since_update > 1 and track.track_id in self.in_frame_tracker:
              logger.debug(f"Face {track.track_id} has exited the frame, removing from in_frame_tracker")
              del self.in_frame_tracker[track.track_id]



    def generate_feature(self, face, frame):
        """
        Generate a feature vector from the detected face region, used for tracking.
        """
        x, y, w, h, _ = face.astype(int)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return np.zeros(128)

        face_roi = cv2.resize(face_roi, (96, 96))  # Resize to fixed size
        return face_roi.flatten() / 255.0  # Normalize the pixel values

    async def match_face(self, embedding):
        """
        Match a face embedding against stored face embeddings.
        """
        selected_faces = await sync_to_async(list)(
            SelectedFace.objects.filter(user=self.user)
        )

        for face in selected_faces:
            if face.embedding:
                distance = np.linalg.norm(np.array(embedding) - np.array(face.embedding))
                if distance < self.face_match_threshold:
                    return face  # Return the matched SelectedFace object
        return None

    async def save_face_image(self, frame, track):
      track_id = int(track.track_id)

      # Only store face if it's not already in the frame
      if track_id in self.in_frame_tracker:
          return None  # Skip processing if the face is already in the frame

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
          embedding = self.generate_face_embedding(face_img)
          if embedding is not None:
              embedding = embedding.tolist()  # Convert numpy array to list for storage
              try:
                  # Encode the face image as a byte array
                  face_img = cv2.imencode('.jpg', face_img)[1].tobytes()

                  # Store face in TempFace model for later processing
                  temp_face = await sync_to_async(TempFace.objects.create)(
                      user=self.user,
                      face_id=face_id,
                      image_data=face_img,
                      embedding=embedding,
                      last_seen=timezone.now(),
                      processed=False
                  )
                  logger.info(f"Temporary face {face_id} saved to TempFace model")
                  return temp_face
              except Exception as e:
                  logger.error(f"Error saving TempFace {face_id}: {str(e)}", exc_info=True)
                  return None
      return None


    def get_next_face_id(self):
        if self.available_face_ids:
            return self.available_face_ids.pop(0)

        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.face_id_counter = 1
            self.face_id_mapping.clear()

        face_id = f"unknown_{self.face_id_counter:03d}"
        self.face_id_counter += 1
        logger.info(f"Generated new face ID: {face_id}")
        return face_id

    def generate_face_embedding(self, face_image):
        # Convert to RGB as face_recognition works with RGB images
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            return encodings[0]
        return None

    async def periodic_processing(self):
        while True:
            try:
                logger.info("Starting periodic processing of temp faces...")
                await self.process_temp_faces()
                logger.info("Finished periodic processing of temp faces")
            except Exception as e:
                logger.error(f"Error in periodic processing: {str(e)}", exc_info=True)
            finally:
                await asyncio.sleep(PROCESSING_INTERVAL)  # Wait for the defined interval before running again

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

      best_image, best_quality_score, best_embedding = None, -float('inf'), None
      last_seen = face_group[0].last_seen

      # Find the best quality face in the group
      for face in face_group:
          image_data = await sync_to_async(lambda: face.image_data)()
          embedding = await sync_to_async(lambda: face.embedding)()
          if image_data and embedding:
              image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

              if image is None:
                  continue

              # Compute the blur and angle scores
              blur_score = self.detect_blur(image)
              angle_score = self.calculate_face_angle(image)

              # Calculate quality score
              quality_score = blur_score - (angle_score / 10)

              if quality_score > best_quality_score:
                  best_quality_score = quality_score
                  best_image = image_data
                  best_embedding = embedding
                  last_seen = face.last_seen

      # Once we have the best image and embedding, check if it matches an existing face
      if best_image is not None and best_embedding is not None:
          matched_face = await self.match_face(best_embedding)

          if matched_face:
              logger.info(f"Matched face_id: {matched_face.face_id}")

              # Update the existing SelectedFace with the latest info
              await self.create_update_selected_face(matched_face.face_id, best_image, best_embedding, best_quality_score,last_seen)

              # Log the visit in FaceVisit model
              await self.log_face_visit(matched_face, best_image, last_seen)

          else:
              # If no match is found, create a new SelectedFace entry
              logger.info(f"No match found, creating new SelectedFace for face_id: {face_id}")
              new_face = await self.create_update_selected_face(face_id,best_image, best_embedding, best_quality_score,last_seen)

              # Log the first visit in FaceVisit model
              await self.log_face_visit(new_face, best_image, last_seen)

          # Store face details and image by date
          #await self.store_face_by_date(face_id, best_image, last_seen)

      # Delete all TempFace records after processing
      await sync_to_async(TempFace.objects.filter(face_id=face_id).delete)()



    def detect_blur(self, image):
        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Laplacian variance to measure blur
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def calculate_face_angle(self, image):
        # Use Haar cascade to detect faces and calculate the face's angle
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
        return 180  # If no face detected, return the worst possible angle

    def detect_faces(self, frame):
        # Convert BGR to RGB (YOLO works on RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use YOLO model to detect faces
        results = self.facemodel(frame_rgb, conf=0.3)  # Confidence threshold to detect faces

        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                faces.append([x1, y1, x2 - x1, y2 - y1, confidence])  # Bounding box with confidence score

        logger.info(f"Detected {len(faces)} faces")
        return np.array(faces)

    async def create_update_selected_face(self, face_id, image_data, embedding, quality_score, last_seen):
        try:
            logger.info(f"Updating/Creating SelectedFace for face_id: {face_id}")

            # Convert last_seen to the correct timezone
            last_seen = last_seen.astimezone(IST)
            date_seen = last_seen.date()

            # Fetch or create SelectedFace entry
            selected_face, created = await sync_to_async(SelectedFace.objects.get_or_create)(
                user=self.user,
                face_id=face_id,
                date_seen=date_seen,
                defaults={
                    'image_data': image_data,
                    'embedding': embedding,
                    'quality_score': quality_score,
                    'last_seen': last_seen
                }
            )

            if not created:
                # Update the existing face with the latest information
                selected_face.image_data = image_data
                selected_face.embedding = embedding
                selected_face.last_seen = last_seen
                selected_face.quality_score = quality_score
                await sync_to_async(selected_face.save)()

            # Send notification for the face
            await self.send_notification(face_id, last_seen, image_data)
    
            return selected_face
        except Exception as e:
            logger.error(f"Error updating/creating face {face_id}: {str(e)}")
      
    
    async def log_face_visit(self, selected_face, image_data, detected_time):
        try:
            logger.info(f"Logging visit for face_id: {selected_face.face_id}")

            # Convert detected_time to the correct timezone
            detected_time = detected_time.astimezone(IST)
            date_seen = detected_time.date()

            # Create a new FaceVisit entry for each detection
            face_visit = await sync_to_async(FaceVisit.objects.create)(
                selected_face=selected_face,
                image_data=image_data,
                detected_time=detected_time,
                date_seen=date_seen
            )
            logger.info(f"Logged FaceVisit for face_id: {selected_face.face_id}, date_seen: {date_seen}")
        except Exception as e:
            logger.error(f"Error logging FaceVisit for face_id {selected_face.face_id}: {str(e)}")


    


    async def send_notification(self, face_id, last_seen, encoded_image_data):
        """
        Send notifications using the encoded image data (base64) without decoding it for sending.
        Decode it when storing the image data in NotificationLog to match binary storage expectations.
        """
        try:
            logger.info(f"Sending notification for face_id {face_id}...")
            
            # Convert `last_seen` to IST
            last_seen_ist = last_seen.astimezone(IST)

            # Format the `last_seen` to a readable format (12-hour with AM/PM)
            formatted_last_seen = last_seen_ist.strftime('%I:%M %p')

            # If the data is binary (bytes), encode it in base64 for WebSocket communication
            if isinstance(encoded_image_data, bytes):
                encoded_image_data = base64.b64encode(encoded_image_data).decode('utf-8')

            # Create notification payload (send encoded image directly)
            notification_data = {
                'face_id': face_id,
                'camera_name': self.camera_name,  # Dynamic camera name can be used
                'detected_time': formatted_last_seen, 
                'image_data': encoded_image_data  # Send base64-encoded image data
            }

            # Send WebSocket notification using base64-encoded image
            channel_layer = get_channel_layer()

            if channel_layer:
                await channel_layer.group_send(
                    f"notifications_{self.user.id}",
                    {
                        'type': 'send_notification',
                        'message': notification_data
                    }
                )
            else:
                logger.error(f"WebSocket connection closed. Unable to send notification for face_id {face_id}.")
                return

            # Decode base64 image data to binary for storing in the database
            image_bytes = base64.b64decode(encoded_image_data)

            # Log notification in the database with binary image data
            await sync_to_async(NotificationLog.objects.create)(
                user=self.user,
                face_id=face_id,
                camera_name=self.camera_name,  # Replace with actual camera name
                detected_time= last_seen_ist,
                notification_sent=True,
                image_data=image_bytes  # Store decoded binary image data
            )

            logger.info(f"Notification sent for face_id {face_id}")
        except Exception as e:
            logger.error(f"Error sending notification for face_id {face_id}: {str(e)}", exc_info=True)

    async def rename_face(self, old_face_id, new_face_id):
        try:
            selected_faces = await sync_to_async(list)(
                SelectedFace.objects.filter(user=self.user, face_id=old_face_id)
            )

            if not selected_faces:
                logger.error(f"No SelectedFace found with face_id {old_face_id}")
                return

            for face in selected_faces:
                face.face_id = new_face_id
                face.is_known = True
                await sync_to_async(face.save)()

            self.available_face_ids.append(old_face_id)

            logger.info(f"Renamed face_id from {old_face_id} to {new_face_id} and marked as known")

        except Exception as e:
            logger.error(f"Error renaming face_id: {str(e)}", exc_info=True)
    
    def get_face_analytics(self):
      try:
          logger.info("Calculating face analytics based on face visits...")
          now = timezone.now()
          today = now.date()

          # Define time periods for analytics
          periods = {
              'today': (today, today + timedelta(days=1)),
              'week': (today - timedelta(days=7), today + timedelta(days=1)),
              'month': (today - timedelta(days=30), today + timedelta(days=1)),
              'year': (today - timedelta(days=365), today + timedelta(days=1))
          }

          # Query all FaceVisit records for the user
          total_visits_query = FaceVisit.objects.filter(selected_face__user=self.user)

          # Calculate distinct face counts based on face_id (not individual visits)
          total_known_faces = (
              total_visits_query.filter(selected_face__is_known=True)
              .values('selected_face__face_id')  # Get face IDs
              .distinct()  # Ensure only unique face IDs
              .count()  # Count distinct known faces
          )
        
          total_unknown_faces = (
              total_visits_query.filter(selected_face__is_known=False)
              .values('selected_face__face_id')
              .distinct()
              .count()  # Count distinct unknown faces
          )

          # Calculate total unique faces (known + unknown)
          total_faces = total_known_faces + total_unknown_faces

          # Period-specific known faces (based on visits as usual)
          known_faces_today = total_visits_query.filter(
              selected_face__is_known=True,
              detected_time__range=(periods['today'][0], periods['today'][1])
          ).count()

          known_faces_week = total_visits_query.filter(
              selected_face__is_known=True,
              detected_time__range=(periods['week'][0], periods['week'][1])
          ).count()

          known_faces_month = total_visits_query.filter(
              selected_face__is_known=True,
              detected_time__range=(periods['month'][0], periods['month'][1])
          ).count()

          known_faces_year = total_visits_query.filter(
              selected_face__is_known=True,
              detected_time__range=(periods['year'][0], periods['year'][1])
          ).count()

          # Facecount for known persons (cumulative count)
          facecount_query = total_visits_query.filter(selected_face__is_known=True)
          facecount = (
              facecount_query.values('selected_face__face_id')
              .annotate(count=Count('id'))
              .order_by('-count')
          )

          # Format the facecount result into a dictionary
          facecount_dict = {fc['selected_face__face_id']: fc['count'] for fc in facecount}
 
          # Construct the analytics response
          analytics = {
              'date': today.isoformat(),
              'total_faces': total_faces,  # Total distinct known + unknown faces
              'known_faces': total_known_faces,  # Distinct known faces
              'unknown_faces': total_unknown_faces,  # Distinct unknown faces
              'known_faces_today': known_faces_today,
              'known_faces_week': known_faces_week,
              'known_faces_month': known_faces_month,
              'known_faces_year': known_faces_year,
              'facecount': facecount_dict  # Cumulative count for known faces
          }

          logger.info(f"Face analytics calculated: {analytics}")
          return analytics

      except Exception as e:
          logger.error(f"Error getting face analytics: {str(e)}")
          return None


