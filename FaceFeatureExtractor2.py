import time
from dataclasses import dataclass
import numpy as np
from head_pose_utils import calculate_head_pose
from stability_utils import calculate_stability
from mouth_utils import calculate_mouth_aspect_ratio
from eyes_utils import (
    calculate_gaze_with_iris,
    calculate_eye_contact,
    calculate_gaze_variation,
    calculate_blinking_ratio,
    EyeContactBuffer
)

# Thresholds for various features
MAR_THRESHOLD = 0.65
MOVEMENT_THRESHOLD = 7
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
GAZE_THRESHOLD_X = 0.1
GAZE_THRESHOLD_Y = 0.6
HISTORY_WINDOW = 15
BLINK_RATIO_THRESHOLD = 4.5
DISTRACTION_TIME_LIMIT = 4

@dataclass
class FaceFeatures:
    head_pitch: float
    head_yaw: float
    head_roll: float
    gaze_x: float
    gaze_y: float
    eye_contact_duration: float
    gaze_variation_x: float
    gaze_variation_y: float
    face_confidence: float
    landmarks_stability: float
    time_since_head_movement: float
    time_since_gaze_shift: float
    mar: float
    blink_ratio: float
    is_blinking: bool
    is_focused: bool
    distraction_duration: float
    eye_contact_detected: bool
    yawn_detected: bool
    is_confused: bool  # Added for confusion
    is_bored: bool     # Added for boredom
    emotion_confidence: float  # Added for emotion confidence

class FaceFeatureExtractor:
    def __init__(self):
        self.eye_contact_buffer = EyeContactBuffer()
        self.gaze_positions_x = []
        self.gaze_positions_y = []
        self.prev_landmarks = None
        self.last_head_movement_time = time.time()
        self.last_gaze_shift_time = time.time()
        self.last_pitch, self.last_yaw, self.last_roll = 0, 0, 0
        self.last_gaze_x, self.last_gaze_y = 0, 0
        self.yawn_detected_time = None

    def extract_features(self, frame, face_landmarks, emotions=None) -> FaceFeatures:
        frame_height, frame_width, _ = frame.shape

        head_pitch, head_yaw, head_roll = calculate_head_pose(face_landmarks, frame)
        gaze_x, gaze_y = calculate_gaze_with_iris(
            face_landmarks.landmark, frame_width, frame_height, True
        )

        if abs(head_pitch - self.last_pitch) > 2 or abs(head_yaw - self.last_yaw) > 2 or abs(head_roll - self.last_roll) > 2:
            self.last_head_movement_time = time.time()
        time_since_head_movement = time.time() - self.last_head_movement_time

        if abs(gaze_x - self.last_gaze_x) > 0.05 or abs(gaze_y - self.last_gaze_y) > 0.05:
            self.last_gaze_shift_time = time.time()
        time_since_gaze_shift = time.time() - self.last_gaze_shift_time

        stability = calculate_stability(face_landmarks, self.prev_landmarks, frame_width, frame_height)
        mar = calculate_mouth_aspect_ratio(face_landmarks.landmark, frame_width, frame_height)

        yawn_detected = False
        if mar > MAR_THRESHOLD:
            if self.yawn_detected_time is None:
                self.yawn_detected_time = time.time()
        else:
            self.yawn_detected_time = None

        if self.yawn_detected_time and time.time() - self.yawn_detected_time <= 2:
            yawn_detected = True

        self.gaze_positions_x.append(gaze_x)
        self.gaze_positions_y.append(gaze_y)
        if len(self.gaze_positions_x) > 100:
            self.gaze_positions_x.pop(0)
            self.gaze_positions_y.pop(0)

        gaze_variation_x, gaze_variation_y = calculate_gaze_variation(self.gaze_positions_x, self.gaze_positions_y)
        self.last_pitch, self.last_yaw, self.last_roll = head_pitch, head_yaw, head_roll
        self.last_gaze_x, self.last_gaze_y = gaze_x, gaze_y
        face_confidence = face_landmarks.landmark[1].visibility
        eye_contact_detected = calculate_eye_contact(gaze_x, gaze_y)
        is_focused, eye_contact_duration, distraction_duration = self.eye_contact_buffer.update_eye_contact(
            eye_contact_detected
        )

        left_blink_ratio = calculate_blinking_ratio(face_landmarks.landmark, LEFT_EYE_POINTS)
        right_blink_ratio = calculate_blinking_ratio(face_landmarks.landmark, RIGHT_EYE_POINTS)
        if left_blink_ratio and right_blink_ratio:
            blink_ratio = (left_blink_ratio + right_blink_ratio) / 2
            is_blinking = blink_ratio > BLINK_RATIO_THRESHOLD
        else:
            blink_ratio = 0.0
            is_blinking = False

        # Emotion detection processing
        is_confused = False
        is_bored = False
        emotion_confidence = 0.0
        if emotions:
            top_emotion = emotions[0]['emotions']
            dominant_emotion = max(top_emotion, key=top_emotion.get)
            emotion_confidence = top_emotion[dominant_emotion]
            is_confused = dominant_emotion in ['surprise'] or top_emotion['surprise'] > 0.3
            is_bored = dominant_emotion in ['neutral', 'sad'] and emotion_confidence > 0.5

        self.prev_landmarks = face_landmarks

        return FaceFeatures(
            head_pitch=head_pitch,
            head_yaw=head_yaw,
            head_roll=head_roll,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            eye_contact_duration=eye_contact_duration,
            gaze_variation_x=gaze_variation_x,
            gaze_variation_y=gaze_variation_y,
            face_confidence=face_confidence,
            landmarks_stability=stability,
            time_since_head_movement=time_since_head_movement,
            time_since_gaze_shift=time_since_gaze_shift,
            mar=mar,
            blink_ratio=blink_ratio,
            is_blinking=is_blinking,
            is_focused=is_focused,
            distraction_duration=distraction_duration,
            eye_contact_detected=eye_contact_detected,
            yawn_detected=yawn_detected,
            is_confused=is_confused,
            is_bored=is_bored,
            emotion_confidence=emotion_confidence
        )