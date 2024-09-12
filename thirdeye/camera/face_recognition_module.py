# camera/face_recognition.py
import base64
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
from .models import TempFace, SelectedFace,NotificationLog
import face_recognition
from django.db.models import Count, Q
from datetime import timedelta
from channels.layers import get_channel_layer
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
FACE_MATCH_THRESHOLD = 0.6

class FaceRecognitionProcessor:
    def __init__(self, user=None,camera_name=None):
        self.user = user
        self.camera_name=camera_name
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

        # Configuration parameters and variables
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
            # Check if the face is confirmed and still being tracked
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # Face bounding box
            bbox = track.to_tlbr()

            # Track ID used to uniquely identify the face
            track_id = track.track_id

            # Check if this is a new face entering the frame
            if track_id not in self.in_frame_tracker:
                # Process and store this face on entry
                temp_face = await self.save_face_image(frame, track)

                if temp_face is None:
                    continue

                # Mark the face as currently in the frame
                self.in_frame_tracker[track_id] = True

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
                logger.debug(f"Stored new face: {temp_face.face_id}")

        # Remove any face that is no longer in the frame (based on tracker status)
        self.cleanup_exited_faces()

        return frame, detected_faces

    def cleanup_exited_faces(self):
        # Remove tracks for faces that have left the frame
        for track in self.tracker.tracks:
            if track.time_since_update > 1 and track.track_id in self.in_frame_tracker:
                # Mark the face as having exited the frame
                logger.debug(f"Face {track.track_id} has exited the frame")
                del self.in_frame_tracker[track.track_id]

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

                    # Check if this face already exists in the database (matching by embedding)
                    matched_face = await self.match_face(embedding)
                    
                    if matched_face:
                        # Use the matched face_id if a match is found
                        face_id = matched_face.face_id
                        await self.create_update_selected_face(face_id, face_img, embedding, 0, timezone.now())
                    else:
                        # Create a new entry if no match is found
                        await self.create_update_selected_face(face_id, face_img, embedding, 0, timezone.now())
                    
                    logger.info(f"Processed and stored face for {face_id}")
                    return TempFace(user=self.user, face_id=face_id, image_data=face_img, embedding=embedding)
                except Exception as e:
                    logger.error(f"Error processing face {face_id}: {str(e)}", exc_info=True)
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

    async def match_face(self, embedding, threshold=None):
        if threshold is None:
            threshold = self.face_match_threshold
        selected_faces = await sync_to_async(list)(SelectedFace.objects.filter(user=self.user))

        for face in selected_faces:
            if face.embedding:
                # Compare embeddings
                distance = np.linalg.norm(np.array(embedding) - np.array(face.embedding))
                if distance < threshold:
                    logger.info(f"Matched face_id: {face.face_id} with distance {distance}")
                    return face  # Return matched face

        return None  # No match found
    
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

      if best_image is not None and best_embedding is not None:
          matched_face = await self.match_face(best_embedding)
          
          if matched_face:
              logger.info(f"Matched face_id: {matched_face.face_id}")
              await self.create_update_selected_face(matched_face.face_id, best_image, best_embedding, best_quality_score, last_seen)
          else:
              logger.info(f"No match found, creating or updating SelectedFace for face_id: {face_id}")
              await self.create_update_selected_face(face_id, best_image, best_embedding, best_quality_score, last_seen)

      # Delete all TempFace records after processing
      await sync_to_async(TempFace.objects.filter(face_id=face_id).delete)()

    def generate_feature(self, face, frame):
      # Extract face region from the frame
      x, y, w, h, _ = face.astype(int)
      face_roi = frame[y:y+h, x:x+w]
      if face_roi.size == 0:
          return np.zeros(128)  # Return a zero-vector if no face region is detected

      # Resize face to a fixed size (e.g., 96x96)
      face_roi = cv2.resize(face_roi, (96, 96))
      return face_roi.flatten() / 255.0  # Normalize pixel values and flatten the array


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

      logger.info(f"Detected {len(faces)} faces")  # Log the number of faces detected
      return np.array(faces)

    async def create_update_selected_face(self, face_id, image_data, embedding, quality_score, last_seen):
        # Create or update SelectedFace entry
        try:
            logger.info(f"Sending face for last_seen {last_seen}...")
            selected_faces = await sync_to_async(list)(
                SelectedFace.objects.filter(user=self.user, face_id=face_id)
            )
            if selected_faces:
                # Update existing face
                selected_face = selected_faces[0]
                selected_face.image_data = image_data
                selected_face.embedding = embedding
                selected_face.last_seen = last_seen
                selected_face.quality_score = quality_score
                await sync_to_async(selected_face.save)()
            else:
                # Create new face entry
                new_face = SelectedFace(
                    user=self.user,
                    face_id=face_id,
                    image_data=image_data,
                    embedding=embedding,
                    quality_score=quality_score,
                    last_seen=last_seen,
                )
                logger.info(f"Sending face for last_seen {selected_face.last_seen}...")
                await sync_to_async(new_face.save)()

            
            logger.info(f"Face {face_id} processed and saved")
        except Exception as e:
            logger.error(f"Error updating/creating face {face_id}: {str(e)}")
 
       # Send notification for the new face
        logger.info(f"Sending notification for last_seen {selected_face.last_seen}...")
        await self.send_notification(face_id, last_seen, image_data)


    #import base64

    async def send_notification(self, face_id, last_seen, encoded_image_data):
      """
      Send notifications using the encoded image data (base64) without decoding it for sending.
      Decode it when storing the image data in NotificationLog to match binary storage expectations.
      """
      try:
          logger.info(f"Sending notification for face_id {face_id}...")
          logger.info(f"Sending notification for last_seen {last_seen}...")
          # If the data is binary (bytes), encode it in base64 for WebSocket communication
          if isinstance(encoded_image_data, bytes):
              encoded_image_data = base64.b64encode(encoded_image_data).decode('utf-8')
          else:
              logger.warning(f"encoded_image_data is not a bytes object. Skipping re-encoding.")

          # Create notification payload (send encoded image directly)
          notification_data = {
              'face_id': face_id,
              'camera_name': self.camera_name,  # Dynamic camera name can be used
              'detected_time': last_seen.strftime('%I:%M %p'),
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
              detected_time=last_seen,
              notification_sent=True,
              image_data=image_bytes  # Store decoded binary image data
          )

          logger.info(f"Notification sent for face_id {face_id}")
          logger.info(f"Notification sent for last_seen {last_seen}")
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
          logger.info("Calculating face analytics...")
          today = timezone.now().date()

          # Define time periods for analytics
          periods = {
              'today': (today, today + timedelta(days=1)),
              'week': (today - timedelta(days=7), today + timedelta(days=1)),
              'month': (today - timedelta(days=30), today + timedelta(days=1)),
              'year': (today - timedelta(days=365), today + timedelta(days=1)),
              'all': (None, None)  # For all-time data
          }

          analytics = {}
          for period, (start_date, end_date) in periods.items():
              # Base query for known faces
              query = Q(user=self.user, is_known=True)
              
              # If there's a specific start and end date, apply range filtering
              if start_date and end_date:
                  query &= Q(last_seen__range=(start_date, end_date))
 
              # Count known faces for the given time period
              known_faces = SelectedFace.objects.filter(query).count()
              analytics[f'known_faces_{period}'] = known_faces
 
          # Calculate total faces (all time)
          total_faces = SelectedFace.objects.filter(user=self.user).count()
 
          # Calculate unknown faces (all time)
          unknown_faces = SelectedFace.objects.filter(user=self.user, is_known=False).count()
  
          # Get the most common face IDs (all time)
          face_counts = (
              SelectedFace.objects.filter(user=self.user)
              .values('face_id')
              .annotate(count=Count('id'))
              .order_by('-count')
          )

          # Update analytics dictionary with total and unknown face data
          analytics.update({
              'total_faces': total_faces,
              'unknown_faces': unknown_faces,
              'known_faces':known_faces,
              'face_counts': list(face_counts),  # List of face ID counts
              'date': today.isoformat()  # Include today's date for reference
          })
 
          logger.info(f"Face analytics calculated: {analytics}")
          return analytics  # Return the full analytics dictionary

      except Exception as e:
          # Catch exceptions and log the error
          logger.error(f"Error getting face analytics: {str(e)}")
          return None  # Return None in case of an error

