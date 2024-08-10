import io
import logging
import socketserver
from http import server
from threading import Condition, Thread
import cv2
import mediapipe as mp
import numpy as np
import json
from collections import deque
import math

# Initialize MediaPipe Face Mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Queue to store orientation history
orientation_history = deque(maxlen=10)  # Increased history length for better smoothing
no_face_counter = 0

PAGE = """\
<html>
<head>
<title>Face Orientation Detection</title>
<style>
    body, html {
        height: 100%;
        width: 100%;
        margin: 0;
        padding: 0;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: black;
    }
    .stream {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
</style>
</head>
<body>
    <img src="stream.mjpg" class="stream"/>
</body>
</html>
"""

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.orientation = "Unknown"
        self.condition = Condition()
        self.detection_count = 0  # Counter for detections

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/orientation':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'orientation': output.orientation,
                'detection_count': output.detection_count  # Send detection count
            }).encode())
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def calculate_angle(nose_tip, left_eye_inner, right_eye_inner):
    # Calculate the horizontal distance between the eyes
    eye_distance = right_eye_inner.x - left_eye_inner.x
    
    # Calculate the vertical distance between the nose and the eye center
    eye_center_y = (left_eye_inner.y + right_eye_inner.y) / 2
    nose_eye_center_distance = nose_tip.y - eye_center_y
    
    # Calculate the angle using the arctangent function
    angle = math.degrees(math.atan2(nose_eye_center_distance, eye_distance))
    
    return angle

def detect_features(output):
    global no_face_counter
    cap = cv2.VideoCapture(0)  # Default camera index

    if not cap.isOpened():
        logging.error("Error: Could not open video capture.")
        return

    logging.info("Video capture started.")

    while True:
        success, frame = cap.read()
        if not success:
            logging.warning("Failed to capture frame.")
            continue

        # Convert the frame to RGB format for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face mesh estimation
        face_mesh_results = face_mesh.process(frame_rgb)

        face_detected = False  # Track if any face is detected

        if face_mesh_results.multi_face_landmarks:
            face_detected = True
            no_face_counter = 0  # Reset no face counter
            output.detection_count += 1  # Increment detection count
            for face_landmarks in face_mesh_results.multi_face_landmarks:
                # Extract landmarks for orientation calculation
                nose_tip = face_landmarks.landmark[1]
                left_eye_inner = face_landmarks.landmark[133]
                right_eye_inner = face_landmarks.landmark[362]

                # Calculate the average position of the inner eye corners
                eye_center_x = (left_eye_inner.x + right_eye_inner.x) / 2 * frame.shape[1]
                nose_x = nose_tip.x * frame.shape[1]

                # Debug information
                print(f"Nose x: {nose_x}, Eye center x: {eye_center_x}")

                # Use the angle calculation
                angle = calculate_angle(nose_tip, left_eye_inner, right_eye_inner)
                print(f"Angle: {angle}")
                normalized_angle = angle if angle < 0 else 180 - angle

                if nose_x < eye_center_x - 10:
                    orientation = "RIGHT"
                elif nose_x > eye_center_x + 10:
                    orientation = "LEFT"
                else:
                    orientation = "CENTER"

                # Add orientation to the history queue
                orientation_history.append(orientation)
                
                # Calculate the most frequent orientation in the history
                most_common_orientation = max(set(orientation_history), key=orientation_history.count)
                
                output.orientation = most_common_orientation

                # Display the orientation on the frame
                cv2.putText(frame, f'Orientation: {most_common_orientation}', (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            no_face_counter += 1
            output.orientation = "No face detected"
            cv2.putText(frame, 'No face detected', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if ret:
            output.write(jpeg.tobytes())
        else:
            logging.error("Failed to encode frame.")

    cap.release()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    output = StreamingOutput()
    address = ('', 8080)
    server = StreamingServer(address, StreamingHandler)

    detect_thread = Thread(target=detect_features, args=(output,))
    detect_thread.start()
    logging.info("Server started at http://localhost:8080")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped by user.")
    finally:
        server.shutdown()
        detect_thread.join()
