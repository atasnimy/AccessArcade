import cv2
import mediapipe as mp

class NoseDetector:
    pass
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_nose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                nose_tip = landmarks.landmark[1]  # Using landmark 1 as an example for the nose tip
                return (nose_tip.x, nose_tip.y)
        return None
