import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from FaceFeatureExtractor2 import FaceFeatureExtractor
import uuid
from datetime import datetime
from ultralytics import YOLO
import torch
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import tempfile
import os
from dataclasses import dataclass
from typing import List, Dict, Any
import json
import io
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import plotly.io as pio
from fer import FER

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Configure Streamlit page
st.set_page_config(
    page_title="AI-Powered Crowd Engagement Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@dataclass
class EngagementMetrics:
    """Data class to store engagement metrics"""
    timestamp: float
    frame_number: int
    is_focused: bool
    eye_contact: bool
    is_blinking: bool
    yawn_detected: bool
    head_pitch: float
    head_yaw: float
    head_roll: float
    gaze_x: float
    gaze_y: float
    mar: float
    distraction_duration: float
    engagement_score: float
    is_confused: bool
    is_bored: bool
    emotion_confidence: float

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate intersection area
    intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

class EngagementAnalyzer:
    
    def __init__(self):
        self.metrics_history: List[EngagementMetrics] = []
        self.sr_enhancer = None
        self.model = None
        self.face_mesh = None
        self.feature_extractor = None
        self.emotion_detector = None
        self.processed_frames = []
        self.face_tracker = {}  # Dictionary to track faces and their states
        self.frame_count = 0  # Track frame number for continuity
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all required models"""
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=10,
                static_image_mode=False,
                refine_landmarks=True,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35
            )
            
            if os.path.exists('yolov11l-face.pt'):
                self.model = YOLO('yolov11l-face.pt')
            else:
                st.warning("YOLO face detection model not found. Please ensure 'yolov11l-face.pt' is in the current directory.")
            
            self.feature_extractor = FaceFeatureExtractor()
            self.sr_enhancer = ESRGANSuperResolution()
            self.emotion_detector = FER(mtcnn=True)
            
            st.success("✅ All models initialized successfully!")
            
        except Exception as e:
            st.error(f"❌ Error initializing models: {str(e)}")
    
    def calculate_engagement_score(self, features) -> float:
        """Calculate engagement score based on various features"""
        score = 100.0
        
        if not features.is_focused:
            score -= 30
        if not features.eye_contact_detected:
            score -= 20
        if features.yawn_detected:
            score -= 25
        if abs(features.head_pitch) > 30:
            score -= 10
        if abs(features.head_yaw) > 30:
            score -= 10
        if features.is_confused:
            score -= 15
        if features.is_bored:
            score -= 20
        
        return max(0, score)
    
    def process_video(self, video_path: str, progress_callback=None, save_output_video=False) -> Dict[str, Any]:
        """Process video and extract engagement metrics"""
        self.metrics_history = []
        self.processed_frames = []
        self.face_tracker = {}  # Reset face tracker for new video
        self.frame_count = 0
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = None
        if save_output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        face_detected_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            if progress_callback:
                progress_callback(frame_count / total_frames)
            
            processed_frame = frame.copy()
            metrics = self.process_frame(frame, frame_count, timestamp, processed_frame)
            if metrics:
                self.metrics_history.append(metrics)
                face_detected_frames += 1
            
            if save_output_video and out is not None:
                out.write(processed_frame)
        
        cap.release()
        if out is not None:
            out.release()
        
        summary = self.calculate_summary_stats(total_frames, face_detected_frames, fps)
        
        result = {
            'metrics': self.metrics_history,
            'summary': summary,
            'total_frames': total_frames,
            'face_detected_frames': face_detected_frames
        }
        
        if save_output_video and out is not None:
            result['output_video_path'] = output_path
        
        return result
    
    def process_frame(self, frame, frame_number: int, timestamp: float, processed_frame=None) -> EngagementMetrics:
        """Process a single frame and extract metrics for all detected faces"""
        if self.model is None:
            return None
        
        self.frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_rgb)
        metrics = None
        
        # Counters for states
        engaged_count = 0
        distracted_count = 0
        bored_count = 0
        yawn_count = 0
        
        # Get video resolution and calculate font scale based on resolution
        height, width = frame.shape[:2]
        # Calculate font scale based on video resolution (baseline: 720p)
        font_scale = max(0.4, min(1.2, (height / 720.0) * 0.7))
        
        # Track number of people detected in this frame
        person_count = 0
        
        # Current frame's face detections
        current_faces = []
        
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                person_count += 1
                current_faces.append((x1, y1, x2, y2))
                
                enhanced_face = self.sr_enhancer.enhance_face(face_crop, (256, 256))
                enhanced_face_rgb = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2RGB)
                
                # Emotion detection
                emotions = self.emotion_detector.detect_emotions(enhanced_face_rgb)
                is_confused = False
                is_bored = False
                emotion_confidence = 0.0
                
                if emotions:
                    top_emotion = emotions[0]['emotions']
                    dominant_emotion = max(top_emotion, key=top_emotion.get)
                    emotion_confidence = top_emotion[dominant_emotion]
                    is_confused = dominant_emotion in ['surprise'] or (top_emotion['surprise'] > 0.3)
                    is_bored = dominant_emotion in ['neutral', 'sad'] and emotion_confidence > 0.5
                
                face_results = self.face_mesh.process(enhanced_face_rgb)
                
                # Track face using bounding box with a lower IoU threshold for better continuity
                face_id = None
                max_iou = 0
                for fid, (prev_box, _) in self.face_tracker.items():
                    iou = calculate_iou((x1, y1, x2, y2), prev_box)
                    if iou > max_iou and iou > 0.3:  # Lowered IoU threshold from 0.5 to 0.3
                        max_iou = iou
                        face_id = fid
                
                if face_id is None:
                    face_id = str(uuid.uuid4())  # New face ID
                    self.face_tracker[face_id] = [
                        (x1, y1, x2, y2), 
                        {
                            'bored_count': 0, 
                            'last_frame': self.frame_count, 
                            'yawn_count': 0,
                            'yawn_state': 'not_yawning',  # 'not_yawning', 'yawning', 'post_yawn'
                            'yawn_frame_count': 0,
                            'post_yawn_frames': 0,
                            'yawn_confidence': 0.0  # Track confidence for more robust detection
                        }
                    ]
                
                # Update face tracker
                self.face_tracker[face_id][0] = (x1, y1, x2, y2)
                tracker = self.face_tracker[face_id][1]
                
                # Update boredom count
                if is_bored:
                    tracker['bored_count'] = tracker.get('bored_count', 0) + 1
                else:
                    tracker['bored_count'] = 0  # Reset if not bored
                
                tracker['last_frame'] = self.frame_count
                
                # Confirm boredom only after 4 consecutive frames
                confirmed_bored = tracker['bored_count'] >= 4
                
                if face_results.multi_face_landmarks:
                    for face_landmarks in face_results.multi_face_landmarks:
                        if processed_frame is not None:
                            h, w = frame.shape[:2]
                            crop_h, crop_w = face_crop.shape[:2]
                            scale_x = (x2 - x1) / 256
                            scale_y = (y2 - y1) / 256
                            
                            for landmark in face_landmarks.landmark:
                                landmark.x = x1 + (landmark.x * 256) * scale_x
                                landmark.y = y1 + (landmark.y * 256) * scale_y
                            
                            mp_drawing.draw_landmarks(
                                image=processed_frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                            mp_drawing.draw_landmarks(
                                image=processed_frame,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                        
                        features = self.feature_extractor.extract_features(
                            enhanced_face_rgb, face_landmarks, emotions
                        )
                        
                        # Use confirmed boredom for state determination
                        features.is_bored = confirmed_bored
                        
                        # Improved yawn detection logic with confidence threshold
                        current_yawning = features.yawn_detected
                        yawn_confidence = features.mar if hasattr(features, 'mar') else 0.0  # Assume MAR is used for yawn detection
                        tracker['yawn_confidence'] = yawn_confidence
                        
                        # State machine for yawn detection with stricter conditions
                        min_yawn_frames = 5  # Require at least 5 frames of continuous yawning
                        min_yawn_confidence = 0.4  # Require MAR >= 0.4 for a yawn
                        cooldown_frames = 15  # Increased cooldown period
                        
                        if tracker['yawn_state'] == 'not_yawning':
                            if current_yawning and yawn_confidence >= min_yawn_confidence:
                                tracker['yawn_state'] = 'yawning'
                                tracker['yawn_frame_count'] = 1
                            # Only increment yawn_count after min_yawn_frames
                        
                        elif tracker['yawn_state'] == 'yawning':
                            if current_yawning and yawn_confidence >= min_yawn_confidence:
                                tracker['yawn_frame_count'] += 1
                                if tracker['yawn_frame_count'] >= min_yawn_frames:
                                    if tracker['yawn_count'] == 0 or tracker['post_yawn_frames'] >= cooldown_frames:
                                        tracker['yawn_count'] += 1  # Increment only after sufficient frames
                            else:
                                tracker['yawn_state'] = 'post_yawn'
                                tracker['post_yawn_frames'] = 1
                        
                        elif tracker['yawn_state'] == 'post_yawn':
                            if current_yawning and yawn_confidence >= min_yawn_confidence:
                                tracker['yawn_state'] = 'yawning'
                                tracker['yawn_frame_count'] += 1
                            else:
                                tracker['post_yawn_frames'] += 1
                                if tracker['post_yawn_frames'] >= cooldown_frames:
                                    tracker['yawn_state'] = 'not_yawning'
                                    tracker['yawn_frame_count'] = 0
                                    tracker['post_yawn_frames'] = 0
                        
                        # For display purposes, show yawning if currently in yawning state
                        display_yawning = tracker['yawn_state'] == 'yawning' and tracker['yawn_frame_count'] >= min_yawn_frames
                        
                        # Determine state and set bbox color
                        if display_yawning:
                            state = "yawn"
                            color = (255, 0, 0)  # Blue
                        elif features.is_bored:
                            state = "bored"
                            color = (128, 128, 128)  # Gray
                        elif not features.is_focused:
                            state = "distracted"
                            color = (0, 165, 255)  # Orange
                        else:
                            state = "engaged"
                            color = (0, 255, 0)  # Green
                        
                        # Draw bbox with appropriate color
                        if processed_frame is not None:
                            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Increment state counters
                        if state == "yawn" and display_yawning:
                            yawn_count = 1  # Only count one yawn per frame for display
                        elif state == "bored":
                            bored_count += 1
                        elif state == "distracted":
                            distracted_count += 1
                        elif state == "engaged":
                            engaged_count += 1
                        
                        if metrics is None:
                            metrics = EngagementMetrics(
                                timestamp=timestamp,
                                frame_number=frame_number,
                                is_focused=features.is_focused,
                                eye_contact=features.eye_contact_detected,
                                is_blinking=features.is_blinking,
                                yawn_detected=display_yawning,  # Use display_yawning for frame-by-frame metrics
                                head_pitch=features.head_pitch,
                                head_yaw=features.head_yaw,
                                head_roll=features.head_roll,
                                gaze_x=features.gaze_x,
                                gaze_y=features.gaze_y,
                                mar=features.mar,
                                distraction_duration=features.distraction_duration,
                                engagement_score=self.calculate_engagement_score(features),
                                is_confused=features.is_confused,
                                is_bored=features.is_bored,
                                emotion_confidence=emotion_confidence
                            )
                else:
                    # Draw red bbox if face mesh fails
                    if processed_frame is not None:
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Clean up face tracker for faces not seen in recent frames
        self.face_tracker = {
            fid: data for fid, data in self.face_tracker.items()
            if self.frame_count - data[1]['last_frame'] <= 10  # Increased to 10 frames for better tracking
        }
        
        # Adjust font scale based on number of people detected (to prevent overcrowding)
        if person_count > 3:
            font_scale *= 0.8
        elif person_count > 6:
            font_scale *= 0.6
        
        # Ensure minimum font scale for readability
        font_scale = max(0.3, font_scale)
        
        # Calculate thickness based on font scale
        thickness = max(1, int(font_scale * 2))
        
        # Draw summary text with proper scaling
        if processed_frame is not None:
            # Get total yawn count from all tracked faces for display
            total_yawn_sessions = sum(tracker[1].get('yawn_count', 0) for tracker in self.face_tracker.values())
            
            summary_text = f"Engaged: {engaged_count}, Distracted: {distracted_count}, Bored: {bored_count}, Yawns: {total_yawn_sessions}"
            
            # Calculate text size for positioning
            (text_width, text_height), baseline = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Position text with padding from edges
            padding = 10
            text_x = padding
            text_y = text_height + padding
            
            # Draw text with outline for better visibility
            cv2.putText(processed_frame, summary_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)  # Black outline
            cv2.putText(processed_frame, summary_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)  # White text

            # Define legend items (label and corresponding BGR color)
            legend_items = [
                ("yawn", (255, 0, 0)),         # Blue
                ("bored", (128, 128, 128)),    # Gray
                ("distracted", (0, 165, 255)), # Orange
                ("engaged", (0, 255, 0)),      # Green
            ]

            # Start position below the summary text
            legend_x = text_x
            legend_y = text_y + 30  # Adjust as needed

            box_size = 20  # Size of the color box
            spacing = 10   # Space between items

            for label, color in legend_items:
                # Draw color box
                cv2.rectangle(processed_frame, (legend_x, legend_y), (legend_x + box_size, legend_y + box_size), color, -1)

                # Draw label next to box (with outline for visibility)
                cv2.putText(processed_frame, label, (legend_x + box_size + 5, legend_y + box_size - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Black outline
                cv2.putText(processed_frame, label, (legend_x + box_size + 5, legend_y + box_size - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)  # White text

                # Move to the next item horizontally
                legend_x += box_size + 5 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + spacing

        
        return metrics
    
    def calculate_summary_stats(self, total_frames: int, face_detected_frames: int, fps: float) -> Dict[str, Any]:
        """Calculate summary statistics from metrics"""
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame([vars(m) for m in self.metrics_history])
        
        focused_percentage = (df['is_focused'].sum() / len(df)) * 100
        eye_contact_percentage = (df['eye_contact'].sum() / len(df)) * 100
        
        # Get total yawn count from all tracked faces
        yawn_count = sum(tracker[1].get('yawn_count', 0) for tracker in self.face_tracker.values())
        
        blink_count = df['is_blinking'].sum()
        confused_percentage = (df['is_confused'].sum() / len(df)) * 100
        bored_percentage = (df['is_bored'].sum() / len(df)) * 100
        
        avg_engagement = df['engagement_score'].mean()
        
        total_duration = total_frames / fps
        face_detection_rate = (face_detected_frames / total_frames) * 100
        
        avg_distraction_duration = df['distraction_duration'].mean()
        max_distraction_duration = df['distraction_duration'].max()
        
        summary = {
            'total_frames': total_frames,
            'total_duration': total_duration,
            'face_detection_rate': face_detection_rate,
            'focused_percentage': focused_percentage,
            'eye_contact_percentage': eye_contact_percentage,
            'avg_engagement_score': avg_engagement,}

        blink_count = df['is_blinking'].sum()
        confused_percentage = (df['is_confused'].sum() / len(df)) * 100
        bored_percentage = (df['is_bored'].sum() / len(df)) * 100
        
        avg_engagement = df['engagement_score'].mean()
        
        total_duration = total_frames / fps
        face_detection_rate = (face_detected_frames / total_frames) * 100
        
        avg_distraction_duration = df['distraction_duration'].mean()
        max_distraction_duration = df['distraction_duration'].max()
        
        summary = {
            'total_frames': total_frames,
            'total_duration': total_duration,
            'face_detection_rate': face_detection_rate,
            'focused_percentage': focused_percentage,
            'eye_contact_percentage': eye_contact_percentage,
            'avg_engagement_score': avg_engagement,
            'yawn_count': int(yawn_count),
            'blink_count': int(blink_count),
            'confused_percentage': confused_percentage,
            'bored_percentage': bored_percentage,
            'avg_distraction_duration': avg_distraction_duration,
            'max_distraction_duration': max_distraction_duration,
            'head_movement_std': {
                'pitch': df['head_pitch'].std(),
                'yaw': df['head_yaw'].std(),
                'roll': df['head_roll'].std()
            },
            'fps': fps
        }
        
        return convert_numpy_types(summary)

class ESRGANSuperResolution:
    def __init__(self):
        self.use_basic_enhancement = True
        
    def enhance_face(self, face_crop, target_size=(256, 256)):
        if face_crop is None or face_crop.size == 0:
            return face_crop
        
        try:
            h, w = face_crop.shape[:2]
            upscaled = cv2.resize(face_crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(upscaled, -1, kernel)
            if target_size:
                final = cv2.resize(sharpened, target_size, interpolation=cv2.INTER_LANCZOS4)
            else:
                final = sharpened
            return final
        except:
            return cv2.resize(face_crop, target_size, interpolation=cv2.INTER_LANCZOS4)

def create_engagement_charts(metrics_data: List[EngagementMetrics]) -> Dict[str, go.Figure]:
    """Create various charts for engagement analysis"""
    df = pd.DataFrame([vars(m) for m in metrics_data])
    
    charts = {}
    
    fig_engagement = px.line(
        df, x='timestamp', y='engagement_score',
        title='Engagement Score Over Time',
        labels={'timestamp': 'Time (seconds)', 'engagement_score': 'Engagement Score (%)'}
    )
    fig_engagement.add_hline(y=70, line_dash="dash", line_color="orange", 
                           annotation_text="Good Engagement Threshold")
    charts['engagement_timeline'] = fig_engagement
    
    fig_focus = make_subplots(
        rows=4, cols=1,
        subplot_titles=['Focus Status', 'Eye Contact Status', 'Confusion Status', 'Boredom Status'],
        vertical_spacing=0.1
    )
    
    fig_focus.add_trace(
        go.Scatter(x=df['timestamp'], y=df['is_focused'].astype(int),
                  mode='lines', name='Focused', line=dict(color='green')),
        row=1, col=1
    )
    
    fig_focus.add_trace(
        go.Scatter(x=df['timestamp'], y=df['eye_contact'].astype(int),
                  mode='lines', name='Eye Contact', line=dict(color='blue')),
        row=2, col=1
    )
    
    fig_focus.add_trace(
        go.Scatter(x=df['timestamp'], y=df['is_confused'].astype(int),
                  mode='lines', name='Confused', line=dict(color='orange')),
        row=3, col=1
    )
    
    fig_focus.add_trace(
        go.Scatter(x=df['timestamp'], y=df['is_bored'].astype(int),
                  mode='lines', name='Bored', line=dict(color='gray')),
        row=4, col=1
    )
    
    fig_focus.update_layout(title='Focus, Eye Contact, Confusion, and Boredom Timeline')
    charts['focus_timeline'] = fig_focus
    
    head_movements_df = df[['timestamp', 'head_pitch', 'head_yaw', 'head_roll']].copy()
    head_movements_df = head_movements_df.melt(
        id_vars=['timestamp'], 
        value_vars=['head_pitch', 'head_yaw', 'head_roll'],
        var_name='movement_type', 
        value_name='angle'
    )
    
    fig_head = px.line(
        head_movements_df, x='timestamp', y='angle', color='movement_type',
        title='Head Movement Analysis',
        labels={'timestamp': 'Time (seconds)', 'angle': 'Angle (degrees)'}
    )
    charts['head_movement'] = fig_head
    
    fig_dist = px.histogram(
        df, x='engagement_score', nbins=20,
        title='Engagement Score Distribution',
        labels={'engagement_score': 'Engagement Score (%)', 'count': 'Frequency'}
    )
    charts['engagement_distribution'] = fig_dist
    
    return charts

def create_pdf_report(summary, metrics_data, total_frames, charts):
    """Generate a comprehensive PDF report with charts and analysis"""
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    story.append(Paragraph("Face Engagement Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Executive Summary", heading_style))
    
    df = pd.DataFrame([vars(m) for m in metrics_data])
    
    exec_summary = f"""
    This report analyzes face engagement metrics extracted from a {summary['total_duration']:.1f}-second video containing {total_frames:,} frames. 
    The analysis achieved a {summary['face_detection_rate']:.1f}% face detection rate and identified an overall engagement score of {summary['avg_engagement_score']:.1f}%.
    
    Key findings include {summary['focused_percentage']:.1f}% focus rate, {summary['eye_contact_percentage']:.1f}% eye contact rate, 
    {summary['confused_percentage']:.1f}% confusion rate, {summary['bored_percentage']:.1f}% boredom rate,
    and {summary['yawn_count']} instances of yawning. The maximum distraction duration was {summary['max_distraction_duration']:.1f} seconds.
    """
    
    story.append(Paragraph(exec_summary, styles['Normal']))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("Detailed Metrics", heading_style))
    
    metrics_data_table = [
        ['Metric', 'Value', 'Assessment'],
        ['Video Duration', f"{summary['total_duration']:.1f} seconds", 'N/A'],
        ['Face Detection Rate', f"{summary['face_detection_rate']:.1f}%", 'Good' if summary['face_detection_rate'] > 80 else 'Needs Improvement'],
        ['Overall Engagement', f"{summary['avg_engagement_score']:.1f}%", 'Excellent' if summary['avg_engagement_score'] > 80 else 'Good' if summary['avg_engagement_score'] > 60 else 'Needs Improvement'],
        ['Focus Rate', f"{summary['focused_percentage']:.1f}%", 'Good' if summary['focused_percentage'] > 70 else 'Needs Improvement'],
        ['Eye Contact Rate', f"{summary['eye_contact_percentage']:.1f}%", 'Good' if summary['eye_contact_percentage'] > 60 else 'Needs Improvement'],
        ['Confusion Rate', f"{summary['confused_percentage']:.1f}%", 'Good' if summary['confused_percentage'] < 20 else 'High'],
        ['Boredom Rate', f"{summary['bored_percentage']:.1f}%", 'Good' if summary['bored_percentage'] < 20 else 'High'],
        ['Yawn Count', f"{summary['yawn_count']}", 'Normal' if summary['yawn_count'] < 5 else 'High'],
        ['Blink Count', f"{summary['blink_count']}", 'Normal'],
        ['Avg Distraction Duration', f"{summary['avg_distraction_duration']:.1f}s", 'Good' if summary['avg_distraction_duration'] < 2 else 'High'],
        ['Max Distraction Duration', f"{summary['max_distraction_duration']:.1f}s", 'Good' if summary['max_distraction_duration'] < 5 else 'Concerning'],
        ['Head Movement Stability', 'Good' if summary['head_movement_std']['yaw'] < 15 else 'Poor', 'Assessment based on yaw stability']
    ]
    
    table = Table(metrics_data_table)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    story.append(PageBreak())
    
    story.append(Paragraph("Visual Analysis", heading_style))
    
    chart_descriptions = {
        'engagement_timeline': """
        Engagement Timeline Chart:
        This chart displays the engagement score throughout the video duration. The engagement score is calculated based on multiple factors including focus, eye contact, head movement, confusion, boredom, and distraction indicators. The orange dashed line represents the threshold for good engagement (70%). Scores consistently above this line indicate strong engagement, while frequent drops below suggest attention issues.
        """,
        'focus_timeline': """
        Focus, Eye Contact, Confusion, and Boredom Timeline:
        This chart shows focus status, eye contact detection, confusion status, and boredom status over time. Focus status indicates attentiveness, eye contact measures direct gaze, confusion indicates surprise or uncertainty, and boredom reflects neutral or sad expressions. These metrics are crucial for assessing engagement quality.
        """,
        'head_movement': """
        Head Movement Analysis:
        This chart tracks head pitch (up/down), yaw (left/right), and roll (tilt) movements throughout the video. Excessive head movement can indicate restlessness or distraction. Stable head position generally correlates with better focus and engagement.
        """,
        'engagement_distribution': """
        Engagement Score Distribution:
        This histogram shows the distribution of engagement scores across all analyzed frames. A distribution skewed toward higher scores indicates consistently good engagement, while a spread distribution or skew toward lower scores suggests variable or poor engagement patterns.
        """
    }
    
    for chart_name, fig in charts.items():
        img_bytes = pio.to_image(fig, format='png', width=800, height=600)
        img_buffer = io.BytesIO(img_bytes)
        story.append(ReportLabImage(img_buffer, width=6*inch, height=4.5*inch))
        story.append(Spacer(1, 10))
        if chart_name in chart_descriptions:
            story.append(Paragraph(chart_descriptions[chart_name], styles['Normal']))
            story.append(Spacer(1, 20))
    
    story.append(PageBreak())
    
    
    story.append(Spacer(1, 20))
    story.append(Paragraph("Technical Details", heading_style))
    
    tech_details = f"""
    Analysis Parameters:
    • Face Detection: YOLO-based detection with MediaPipe facial landmarks
    • Emotion Detection: FER model with MTCNN
    • Feature Extraction: 468-point facial mesh analysis
    • Engagement Calculation: Multi-factor scoring system including focus, eye contact, head movement, confusion, boredom, and behavioral indicators
    • Processing Rate: {len(df)} frames analyzed out of {total_frames} total frames
    • Temporal Resolution: {summary['fps']:.1f} frames per second
    
    Data Quality:
    • Face Detection Success Rate: {summary['face_detection_rate']:.1f}%
    • Analysis Coverage: {(len(df)/total_frames)*100:.1f}% of video duration
    • Temporal Consistency: Continuous tracking maintained throughout analysis
    """
    
    story.append(Paragraph(tech_details, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    
    return buffer.getvalue()

def main():
    """Main Streamlit application"""
    
    st.title("👁️ Face Engagement Analyzer")
    st.markdown("Upload a video to analyze face engagement, attention, confusion, and boredom metrics using AI-powered computer vision.")
    
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.subheader("Model Status")
    
    if 'analyzer' not in st.session_state:
        with st.spinner("Initializing models..."):
            st.session_state.analyzer = EngagementAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    if analyzer.model is not None:
        st.sidebar.success("✅ YOLO Face Detection")
    else:
        st.sidebar.error("❌ YOLO Face Detection")
    
    if analyzer.face_mesh is not None:
        st.sidebar.success("✅ MediaPipe Face Mesh")
    else:
        st.sidebar.error("❌ MediaPipe Face Mesh")
    
    if analyzer.feature_extractor is not None:
        st.sidebar.success("✅ Feature Extractor")
    else:
        st.sidebar.error("❌ Feature Extractor")
    
    if analyzer.emotion_detector is not None:
        st.sidebar.success("✅ Emotion Detector")
    else:
        st.sidebar.error("❌ Emotion Detector")
    
    st.sidebar.success("✅ Super Resolution")
    
    st.sidebar.subheader("Processing Parameters")
    min_face_size = st.sidebar.slider("Minimum Face Size", 30, 100, 50)
    enhancement_size = st.sidebar.selectbox("Enhancement Size", [128, 256, 512], index=1)
    save_output_video = st.sidebar.checkbox("Generate Output Video with Annotations", value=False)
    
    st.subheader("📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to analyze face engagement metrics"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Duration", f"{duration:.1f}s")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("Total Frames", f"{frame_count:,}")
        
        if st.button("🚀 Analyze Video", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing... {progress*100:.1f}%")
            
            start_time = time.time()
            
            with st.spinner("Analyzing video for engagement metrics..."):
                try:
                    results = analyzer.process_video(video_path, update_progress, save_output_video)
                    processing_time = time.time() - start_time
                    st.success(f"✅ Analysis complete! Processed in {processing_time:.1f} seconds")
                    st.session_state.results = results
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.stop()
        
        if 'results' in st.session_state:
            results = st.session_state.results
            metrics_data = results['metrics']
            summary = results['summary']
            
            if metrics_data:
                st.subheader("📊 Engagement Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Overall Engagement", 
                        f"{summary['avg_engagement_score']:.1f}%",
                        delta=f"{summary['avg_engagement_score'] - 70:.1f}%" if summary['avg_engagement_score'] > 70 else None
                    )
                
                with col2:
                    st.metric(
                        "Focus Rate", 
                        f"{summary['focused_percentage']:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        "Eye Contact Rate", 
                        f"{summary['eye_contact_percentage']:.1f}%"
                    )
                
                with col4:
                    st.metric(
                        "Face Detection Rate", 
                        f"{summary['face_detection_rate']:.1f}%"
                    )
                
                st.subheader("📈 Detailed Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Yawn Count", summary['yawn_count'])
                    st.metric("Avg Distraction Duration", f"{summary['avg_distraction_duration']:.1f}s")
                
                with col2:
                    st.metric("Blink Count", summary['blink_count'])
                    st.metric("Max Distraction Duration", f"{summary['max_distraction_duration']:.1f}s")
                
                with col3:
                    st.metric("Confusion Rate", f"{summary['confused_percentage']:.1f}%")
                    st.metric("Boredom Rate", f"{summary['bored_percentage']:.1f}%")
                
                with col4:
                    st.metric("Head Movement Stability", 
                             "Good" if summary['head_movement_std']['yaw'] < 15 else "Poor")
                
                st.subheader("📊 Engagement Analytics")
                
                charts = create_engagement_charts(metrics_data)
                
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Engagement Timeline", "Focus & Emotion Analysis", "Head Movement", "Score Distribution"
                ])
                
                with tab1:
                    st.plotly_chart(charts['engagement_timeline'], use_container_width=True)
                
                with tab2:
                    st.plotly_chart(charts['focus_timeline'], use_container_width=True)
                
                with tab3:
                    st.plotly_chart(charts['head_movement'], use_container_width=True)
                
                with tab4:
                    st.plotly_chart(charts['engagement_distribution'], use_container_width=True)
                
                st.subheader("💾 Export Data & Results")
                
                df_export = pd.DataFrame([vars(m) for m in metrics_data])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV Data",
                        csv,
                        f"engagement_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv"
                    )
                
                with col2:
                    json_data = json.dumps(summary, indent=2, default=str)
                    st.download_button(
                        "📋 Download Summary JSON",
                        json_data,
                        f"engagement_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                
                with col3:
                    try:
                        pdf_data = create_pdf_report(summary, metrics_data, results['total_frames'], charts)
                        st.download_button(
                            "📄 Download PDF Report",
                            pdf_data,
                            f"engagement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            "application/pdf"
                        )
                    except Exception as e:
                        st.error(f"❌ Failed to generate PDF report: {str(e)}")
                
                with col4:
                    if 'output_video_path' in results and os.path.exists(results['output_video_path']):
                        if os.path.getsize(results['output_video_path']) > 0:
                            with open(results['output_video_path'], 'rb') as f:
                                video_bytes = f.read()
                            st.download_button(
                                label="🎥 Download Analyzed Video",
                                data=video_bytes,
                                file_name=f"analyzed_{uploaded_file.name}",
                                mime="video/mp4"
                            )
                        else:
                            st.warning("⚠️ Analyzed video file is empty.")
                    else:
                        st.warning("⚠️ Analyzed video not available for download.")
                
                with st.expander("🔍 View Raw Data"):
                    st.dataframe(df_export, use_container_width=True)
            
            else:
                st.warning("⚠️ No face detected in the video. Please try with a video containing clear face visibility.")
        
        try:
            os.unlink(video_path)
            if 'results' in st.session_state and 'output_video_path' in st.session_state.results:
                pass
        except:
            pass
    
    else:
        st.info("👆 Please upload a video file to begin analysis")
        
        st.subheader("📋 Features")
        st.markdown("""
        This tool analyzes video content to measure engagement, attention, confusion, and boredom levels using:
        
        - **Face Detection**: YOLO-based face detection with real-time annotation
        - **Super Resolution**: Enhanced face quality for better analysis
        - **Landmark Detection**: 468-point face mesh analysis
        - **Emotion Detection**: FER-based confusion and boredom detection
        - **Engagement Metrics**: Focus, eye contact, head movement, yawning, confusion, and boredom detection
        - **Real-time Analytics**: Comprehensive charts and statistics
        - **Video Output**: Annotated video with engagement and emotion overlays
        - **PDF Reports**: Professional reports with charts and recommendations
        
        **Supported Formats**: MP4, AVI, MOV, MKV
        """)
        
        st.subheader("🔧 Setup Requirements")
        st.markdown("""
        Make sure you have the following dependencies installed:
        ```bash
        pip install streamlit opencv-python mediapipe ultralytics plotly pandas reportlab fer
        Required files in your directory:

        yolov11l-face.pt - YOLO face detection model
        FaceFeatureExtractor.py - Feature extraction module
        Note: The system will automatically handle video processing and generate:

        Annotated output video with engagement and emotion metrics overlay
        Professional PDF report with all charts and analysis
        CSV data export for further analysis """)
        st.subheader("📊 Output Features")
        st.markdown("""
        Analyzed Video Includes:

        Real-time engagement score display
        Focus, eye contact, confusion, and boredom indicators
        Yawn and blink detection alerts
        Head movement tracking
        Timestamp overlay
        PDF Report Contains:

        Executive summary with key findings
        Detailed metrics table with assessments
        All interactive charts with explanations
        Professional recommendations
        Technical analysis details """)

if __name__ == "__main__":
    main()