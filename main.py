from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
  """
  Blurs the face region based on landmarks detected on the faces.
  """
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  for face_landmarks in face_landmarks_list:
    # Find the boundaries of the face region using landmark coordinates
    min_x = min(landmark.x for landmark in face_landmarks)
    max_x = max(landmark.x for landmark in face_landmarks)
    min_y = min(landmark.y for landmark in face_landmarks)
    max_y = max(landmark.y for landmark in face_landmarks)

    # Scale coordinates to image dimensions
    h, w, _ = rgb_image.shape
    # Expand boundaries slightly to ensure full face blur (for privacy/KVKK)
    padding_x = int(w * 0.02)  # Add 2% width padding
    padding_y = int(h * 0.05)  # Add 5% height padding

    x1 = max(0, int(min_x * w) - padding_x)
    y1 = max(0, int(min_y * h) - padding_y)
    x2 = min(w, int(max_x * w) + padding_x)
    y2 = min(h, int(max_y * h) + padding_y)

    # Crop the face region
    face_region = rgb_image[y1:y2, x1:x2]

    # Apply Gaussian blur
    # The blur kernel dimensions (width, height) must be odd and positive.
    # Here, we use a kernel size adjusted dynamically based on the region size.
    # Ensure a minimum kernel size of 11x11.
    ksize_w = max(11, int((x2 - x1) * 0.2)) # 20% of face width (can be increased for more blur)
    ksize_h = max(11, int((y2 - y1) * 0.2)) # 20% of face height (can be increased for more blur)
    # Ensure odd kernel dimensions
    if ksize_w % 2 == 0: ksize_w += 1
    if ksize_h % 2 == 0: ksize_h += 1

    # You can also increase blur by manually setting the SigmaX value (e.g., 15 or 20)
    blurred_face = cv2.GaussianBlur(face_region, (ksize_w, ksize_h), 0) # If SigmaX is 0, it's derived from ksize

    # Place the blurred face region back onto the original image
    annotated_image[y1:y2, x1:x2] = blurred_face

  return annotated_image

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False, # Facial blendshapes are no longer retrieved
                                       output_facial_transformation_matrixes=False, # Transformation matrices are no longer retrieved
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Capture video from camera
cam = cv2.VideoCapture(0)
while cam.isOpened():
    success, frame = cam.read()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # STEP 4: Detect face landmarks from the input image.
        detection_result = detector.detect(mp_image)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("face", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            exit(0)

# Release camera and destroy all windows
cam.release()
cv2.destroyAllWindows()