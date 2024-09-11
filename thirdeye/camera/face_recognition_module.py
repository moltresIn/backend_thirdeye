# camera/face_recognition.py

import cv2
import numpy as np
import os
import base64
import logging
import asyncio
import torch
import face_recognition
from ultralytics import YOLO
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
from django.utils import timezone
from django.conf import settings
from django.db.models import Count
from asgiref.sync import sync_to_async
from channels.layers import get_channel_layer
from datetime import date, timedelta
from .models import TempFace, SelectedFace, NotificationLog

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
notification_logger = logging.getLogger('notifications')

# Configuration parameters
MAX_IMAGES_PER_FACE_ID = 15  # Max 15 records per face ID
FACE_SAVE_INTERVAL = 7  # Save every 7th frame
PROCESSING_INTERVAL = 1  # How often to process faces
MAX_COSINE_DISTANCE = 0.3  # Cosine distance threshold for face matching
NN_BUDGET = 300  # Budget for tracking using DeepSORT
TRACKER_MAX_AGE = 100  # Max age of a track before it's deleted
FACE_MATCH_THRESHOLD = 0.6  # Threshold for face similarity
FACE_REAPPEAR_TIMEOUT = timedelta(seconds=30)  # Timeout before reprocessing a face


class FaceRecognitionProcessor:
    """
    This class handles the detection, tracking, and face recognition process, including saving faces, 
    sending notifications, and avoiding redundant processing.
    """
    def __init__(self, user=None,camera_name=None):
        # Device selection: use GPU if available, else use CPU
        self.user = user
        self.camera_name=camera_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")

        # Load YOLO model for face detection
        model_path = os.path.join(settings.BASE_DIR, 'yolov8m-face.pt')
        self.facemodel = YOLO(model_path).to(self.device)
        logger.info(f"YOLO model loaded on device: {self.device}")

        # Initialize DeepSORT tracker for object tracking (in this case, faces)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
        self.tracker = Tracker(metric, max_age=TRACKER_MAX_AGE)
        logger.info("DeepSORT tracker initialized")

        # Initialize other instance variables
        self.face_match_threshold = FACE_MATCH_THRESHOLD
        self.current_date = date.today()
        self.face_id_counter = 1  # Counter for new face IDs
        self.face_id_mapping = {}  # Mapping of track IDs to face IDs
        self.frame_save_counter = {}  # Counter to save faces at intervals
        self.available_face_ids = []  # List of reusable face IDs
        self.frame_buffer = asyncio.Queue(maxsize=10)  # Buffer to store frames for processing
        self.processed_faces = {}
        self.min_quality_score_threshold = 100
        logger.info("FaceRecognitionProcessor initialized")

        # Start periodic processing of faces (asynchronously)
        self.periodic_task = asyncio.create_task(self.periodic_processing())
        logger.info("Periodic processing task started")

    async def process_frame(self, frame):
      """
      This function processes the incoming frame to detect faces.
      It should return the processed frame and the list of detected faces.
      """
      try:
          logger.debug("Processing new frame")
          await self.frame_buffer.put(frame)  # Add frame to the buffer
        
          processed_frame, detected_faces = await self.process_frame_from_buffer()

          # Ensure something is returned
          if processed_frame is None or detected_faces is None:
              logger.error("Error: process_frame_from_buffer returned None")
              return frame, []  # If something goes wrong, return the original frame and an empty face list

          return processed_frame, detected_faces

      except Exception as e:
          logger.error(f"Error in process_frame: {str(e)}", exc_info=True)
          return frame, []  # Return the original frame and empty face list if an error occurs


    async def process_frame_from_buffer(self):
      """
      Process the frame and only handle faces when they re-enter the frame.
      """
      try:
          frame = await self.frame_buffer.get()
          faces = self.detect_faces(frame)  # Detect faces
          logger.debug(f"Detected {len(faces)} faces")

          detections = [Detection(face[:4], face[4], self.generate_feature(face, frame)) for face in faces]
          self.tracker.predict()
          self.tracker.update(detections)
          logger.debug(f"Tracker updated with {len(self.tracker.tracks)} tracks")

          # Check which tracks have left the frame
          tracks_out_of_frame = [track_id for track_id in self.processed_faces.keys() if track_id not in [t.track_id for t in self.tracker.tracks]]

          #  Reset processed status for faces that left the frame
          for track_id in tracks_out_of_frame:
              del self.processed_faces[track_id]
              logger.debug(f"Track {track_id} left the frame, resetting processed status.")
 
          detected_faces = []
          for track in self.tracker.tracks:
              if not track.is_confirmed() or track.time_since_update > 1:
                  continue  # Skip unconfirmed or outdated tracks

              bbox = track.to_tlbr()
              temp_face = await self.save_face_image(frame, track)  # Save each face

              if temp_face is None:
                  continue

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

          return frame, detected_faces

      except Exception as e:
          logger.error(f"Error in process_frame_from_buffer: {str(e)}", exc_info=True)
          return None, None


  

    def detect_faces(self, frame):
        """
        This function detects faces in the given frame using the YOLO model.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = self.facemodel(frame_rgb, conf=0.3)  # Detect faces with a 0.3 confidence threshold

        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                faces.append([x1, y1, x2 - x1, y2 - y1, confidence])

        logger.info(f"Detected faces: {len(faces)}")
        return np.array(faces)  # Return the list of detected faces

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

    def get_next_face_id(self):
        """
        Generate a new face ID if needed, or reuse an available face ID.
        """
        if self.available_face_ids:
            return self.available_face_ids.pop(0)

        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.face_id_counter = 1  # Reset counter every day
            self.face_id_mapping.clear()
  
        face_id = f"unknown_{self.face_id_counter:03d}"  # Generate a new face ID
        self.face_id_counter += 1
        logger.info(f"Generated new face ID: {face_id}")
        return face_id

    #import base64

    async def save_face_image(self, frame, track):
      """
      Save encoded face images directly and send a notification based on the stored face.
      """
      track_id = int(track.track_id)

      # Skip reprocessing if the face is already stored
      if track_id in self.processed_faces:
          logger.debug(f"Skipping reprocessing for track_id {track_id}, already stored.")
          return None

      if track_id not in self.face_id_mapping:
          self.face_id_mapping[track_id] = self.get_next_face_id()
          self.frame_save_counter[track_id] = 0

      self.frame_save_counter[track_id] += 1
      if self.frame_save_counter[track_id] % FACE_SAVE_INTERVAL != 0:
          return None  # Only save every nth frame

      face_id = self.face_id_mapping[track_id]
      bbox = track.to_tlbr()
      h, w = frame.shape[:2]
      pad_w, pad_h = 0.2 * (bbox[2] - bbox[0]), 0.2 * (bbox[3] - bbox[1])
      x1, y1 = max(0, int(bbox[0] - pad_w)), max(0, int(bbox[1] - pad_h))
      x2, y2 = min(w, int(bbox[2] + pad_w)), min(h, int(bbox[3] + pad_h))

      face_img = frame[y1:y2, x1:x2]  # Extract face region
      if face_img.size > 0:
          # Encode the face image to JPEG format and then base64 for storage
          _, face_encoded = cv2.imencode('.jpg', face_img)  # Convert to JPEG
          face_encoded_base64 = base64.b64encode(face_encoded).decode('utf-8')

          embedding = self.generate_face_embedding(face_img)
          if embedding is not None:
              embedding = embedding.tolist()

              # Calculate quality score
              quality_score = self.calculate_quality_score(face_img)

              # Store the image if it's quality or the limit hasn't been reached
              total_images_stored = await sync_to_async(lambda: SelectedFace.objects.filter(face_id=face_id).count())()
              if total_images_stored >= MAX_IMAGES_PER_FACE_ID:
                  logger.debug(f"Max images stored for face_id {face_id}, skipping further storage.")
                  return None

              if quality_score >= self.min_quality_score_threshold or total_images_stored < MAX_IMAGES_PER_FACE_ID:
                  await self.create_update_selected_face(face_id, face_encoded_base64, embedding, timezone.now())

                  # Mark the face as processed to avoid reprocessing
                  self.processed_faces[track_id] = True
                  logger.info(f"Stored encoded face {face_id}. Notification will be triggered.")


 

 

    #import base64

    async def create_update_selected_face(self, face_id, encoded_image_data, embedding, last_seen):
      """
      Create or update a SelectedFace entry with encoded image data (base64).
      If an entry already exists for the same face_id, user, and date_seen, update it with the new last_seen time.
      """
      try:
          logger.info(f"Start processing face_id {face_id} for user {self.user}")
  
          # Convert last_seen to date for uniqueness checking
          date_seen = last_seen.date()
 
          # Check if a record already exists for this face_id, user, and date_seen
          existing_face = await sync_to_async(lambda: SelectedFace.objects.filter(
              user=self.user, face_id=face_id, date_seen=date_seen
          ).first())()

          # Decode the base64-encoded image back into bytes before saving
          image_bytes = base64.b64decode(encoded_image_data)
 
          if existing_face:
              # Update the existing record if it exists
              existing_face.last_seen = last_seen  # Update the last seen timestamp
              existing_face.image_data = image_bytes  # Optionally update the image data
              existing_face.embedding = embedding  # Optionally update the embedding
              await sync_to_async(existing_face.save)()  # Save changes
              logger.info(f"Updated existing face record for face_id {face_id} at {last_seen}")
          else:
              # Create a new record if no existing entry is found
              new_face = SelectedFace(
                  face_id=face_id,
                  user=self.user,
                  image_data=image_bytes,  # Store as binary data (bytes)
                  embedding=embedding,
                  last_seen=last_seen,
                  date_seen=date_seen  # Store the date for uniqueness constraint
              )
              await sync_to_async(new_face.save)()  # Save the face entry in the database
              logger.info(f"Stored new face entry for face_id {face_id} at {last_seen}")

          # Send notification for the new or updated face
          await self.send_notification(face_id, last_seen, encoded_image_data)

      except Exception as e:
          logger.error(f"Error creating/updating SelectedFace for {face_id}: {str(e)}", exc_info=True)




  

  
   

    def generate_face_embedding(self, face_image):
        """
        Generate face embedding using the face_recognition library.
        """
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if encodings:
            return encodings[0]
        return None

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

    async def periodic_processing(self):
        """
        Periodically process faces stored in the TempFace model and move them to SelectedFace.
        """
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
        """
        Process TempFace objects, move them to SelectedFace, and mark them as processed.
        """
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
        """
        Process a group of faces with the same face_id and move them to SelectedFace.
        """
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

                blur_score = self.detect_blur(image)
                quality_score = blur_score  # Additional quality checks could be added

                if quality_score > best_quality_score:
                    best_quality_score = quality_score
                    best_image = image_data
                    best_embedding = embedding
                    last_seen = face.last_seen

        if best_image is not None and best_embedding is not None:
            matched_face = await self.match_face(best_embedding)
            if matched_face:
                logger.info(f"Matched face_id: {matched_face.face_id}")
                await self.create_update_selected_face(matched_face.face_id, best_image, best_embedding, last_seen)
            else:
                logger.info(f"No match found, creating or updating SelectedFace for face_id: {face_id}")
                await self.create_update_selected_face(face_id, best_image, best_embedding, last_seen)

        # Mark the processed faces as completed
        await sync_to_async(TempFace.objects.filter(face_id=face_id).update)(processed=True)
        logger.info(f"Marked TempFace records for face_id {face_id} as processed")


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

    #import base64
    #import numpy as np

    #import base64

    async def send_notification(self, face_id, last_seen, encoded_image_data):
      """
      Send notifications using the encoded image data (base64) without decoding it for sending.
      Decode it when storing the image data in NotificationLog to match binary storage expectations.
      """
      try:
          notification_logger.info(f"Sending notification for face_id {face_id}...")

          # Create notification payload (send encoded image directly)
          notification_data = {
              'face_id': face_id,
              'camera_name': self.camera_name,  # Dynamic camera name can be used
              'detected_time': last_seen.strftime('%I:%M %p'),
              'image_data': encoded_image_data  # Send base64-encoded image data
          }

          # Send WebSocket notification using base64-encoded image
          channel_layer = get_channel_layer()
          await channel_layer.group_send(
              f"notifications_{self.user.id}",
              {
                  'type': 'send_notification',
                  'message': notification_data
              }
          )

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
  
          notification_logger.info(f"Notification sent for face_id {face_id}")

      except Exception as e:
          notification_logger.error(f"Error sending notification for face_id {face_id}: {str(e)}", exc_info=True)
  

  

    def calculate_quality_score(self, face_img):
        """
        Calculate the quality score of a face image based on the blur detection using Laplacian variance.
        Higher variance means sharper image, lower variance means more blurry.
        """
        gray_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()  # Calculate Laplacian variance
        return laplacian_var  # The higher the variance, the sharper the image
