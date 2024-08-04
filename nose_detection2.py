# nose_detection2.py
import cv2
import mediapipe as mp


class NoseDetector2:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.drawing_utils = mp.solutions.drawing_utils
        self.nose_landmark_index = 1  # Index for the tip of the nose


    def detect_nose(self, frame):
        rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_landmark = face_landmarks.landmark[self.nose_landmark_index]
                nose_x = int(nose_landmark.x * frame.shape[1])
                nose_y = int(nose_landmark.y * frame.shape[0])
                return nose_x, nose_y
        return None, None


    def draw_nose(self, frame, nose_x, nose_y):
        if nose_x is not None and nose_y is not None:
            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)
        return frame
    
