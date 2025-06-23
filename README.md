# BlurGuard: Image Processing Project for GDPR/KVKK Compliant Face Blurring

A real-time face blurring project using MediaPipe and OpenCV to detect faces in live video streams and apply Gaussian blur for enhanced visual privacy. Designed to help meet personal data privacy (KVKK/GDPR) requirements.

## Features

* **Real-time Face Detection:** Faces are detected in live video streams using the MediaPipe Face Landmarker model.
* **Gaussian Blurring:** Gaussian blur effect is applied to detected face regions to ensure visual privacy.
* **Simple and Lightweight:** Focuses solely on core face blurring functionality, without unnecessary complexity.
* **GDPR/KVKK Compliant Design:** Helps protect individual identities, reducing data privacy concerns.

## Prerequisites

To run this project, the following libraries must be installed in your Python environment:

* `opencv-python`
* `mediapipe`
* `numpy`

You can install these libraries using the following commands:

```bash
pip install opencv-python mediapipe numpy
