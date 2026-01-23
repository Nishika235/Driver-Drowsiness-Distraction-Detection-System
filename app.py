import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import logging
import time
import sys
import threading
from collections import deque
from scipy.spatial import distance
from scipy.signal import butter, lfilter, find_peaks
import datetime
import math
from pygame import mixer
import imutils
import pyttsx3
import random

if np.__version__.startswith("2"):
    st.error(f"Incompatible NumPy version detected: {np.__version__}. Please run: pip install \"numpy<2.0\"")
    st.stop()

# --- LOGGING SETUP ---
logging.basicConfig(
    filename='driver_monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- TENSORFLOW SETUP ---
try:
    import tensorflow as tf
    try:
        from tensorflow import keras
    except ImportError:
        import keras
except ImportError:
    st.error("TensorFlow not found. Please run: pip install -r requirements.txt")
    st.stop()

# --- PAGE CONFIG ---
st.set_page_config(page_title="Driver Drowsiness & Distraction Detection System", layout="wide")
st.title("🚗 Driver Drowsiness & Distraction Detection System")

# --- SESSION STATE INITIALIZATION ---
if 'ear_thresh' not in st.session_state: st.session_state.ear_thresh = 0.21
if 'mar_thresh' not in st.session_state: st.session_state.mar_thresh = 0.65
if 'calibrated' not in st.session_state: st.session_state.calibrated = False

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Sensitivity Settings")
st.sidebar.info("Lower values = More sensitive (more alerts)")

EAR_THRESH = st.sidebar.slider("Eye Aspect Ratio (EAR) Threshold", 0.15, 0.35, key='ear_thresh', help="Lower = more sensitive to drowsiness")
MAR_THRESH = st.sidebar.slider("Mouth Aspect Ratio (MAR) Threshold", 0.4, 0.9, key='mar_thresh', help="Lower = more sensitive to yawning")
FRAME_CHECK = st.sidebar.slider("Alert Duration (frames)", 20, 80, 40, help="Higher = fewer false alarms")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Advanced Thresholds")

YAW_WARN = st.sidebar.slider("Head Turn Warning (degrees)", 25, 60, 40)
YAW_CRITICAL = st.sidebar.slider("Head Turn Critical (degrees)", 40, 80, 55)
PITCH_WARN = st.sidebar.slider("Head Tilt Warning (degrees)", 20, 50, 35)
PITCH_CRITICAL = st.sidebar.slider("Head Tilt Critical (degrees)", 35, 70, 50)

PERCLOS_WARN = st.sidebar.slider("PERCLOS Warning (%)", 15, 35, 25) / 100
PERCLOS_CRITICAL = st.sidebar.slider("PERCLOS Critical (%)", 25, 50, 35) / 100

st.sidebar.markdown("---")

# --- CALIBRATION LOGIC ---
if st.sidebar.button("🎯 Calibrate Thresholds (3s)", help="Personalize detection to your face"):
    st.sidebar.info("📸 Keep eyes open, mouth closed, and look straight ahead...")
    cal_cap = cv2.VideoCapture(0)
    ear_list, mar_list = [], []
    
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as cal_mesh:
        for i in range(90):  # 3 seconds at ~30fps
            ret, frame = cal_cap.read()
            if ret:
                frame = imutils.resize(frame, width=480)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = cal_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0]
                    h, w, c = frame.shape
                    shape = np.array([[int(l.x * w), int(l.y * h)] for l in lm.landmark])
                    ear_list.append((eye_aspect_ratio(shape[[33, 160, 158, 133, 153, 144]]) + 
                                     eye_aspect_ratio(shape[[362, 385, 387, 263, 373, 380]])) / 2.0)
                    mar_list.append(mouth_aspect_ratio(shape))
    cal_cap.release()
    
    if ear_list and len(ear_list) > 20:
        # Use median for robustness
        calibrated_ear = np.median(ear_list)
        calibrated_mar = np.median(mar_list)
        
        st.session_state.ear_thresh = round(calibrated_ear - 0.05, 3)  # 5% below resting
        st.session_state.mar_thresh = round(calibrated_mar + 0.20, 2)  # 20% above resting
        st.session_state.calibrated = True
        
        st.sidebar.success(f"✅ Calibrated! EAR: {st.session_state.ear_thresh:.3f}, MAR: {st.session_state.mar_thresh:.2f}")
        time.sleep(2)
        st.rerun()
    else:
        st.sidebar.error("❌ Calibration failed - no face detected")

if st.sidebar.button("🔄 Reset Monitor"):
    st.session_state.calibrated = False
    st.rerun()

# --- HELPER FUNCTIONS ---
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(shape):
    ver = distance.euclidean(shape[13], shape[14])
    hor = distance.euclidean(shape[78], shape[308])
    return ver / hor

def get_head_pose(shape):
    """Calculate head pose - more accurate method"""
    nose = shape[1]
    left_eye = shape[33]
    right_eye = shape[263]
    chin = shape[152]
    
    # Yaw (horizontal turn)
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    nose_to_center = nose[0] - eye_center[0]
    eye_width = abs(right_eye[0] - left_eye[0])
    yaw = (nose_to_center / max(eye_width, 1)) * 90
    
    # Pitch (vertical tilt)
    dy = chin[1] - nose[1]
    dx = chin[0] - nose[0]
    pitch = math.degrees(math.atan2(dy, dx)) - 90
    
    return yaw, pitch

def preprocess_eye(frame, eye_points):
    try:
        (x, y, w, h) = cv2.boundingRect(eye_points)
        pad = 5
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = w + 2 * pad, h + 2 * pad
        eye_img = frame[y:y+h, x:x+w]
        if eye_img.size == 0: return None
        eye_img = cv2.resize(eye_img, (24, 24))
        eye_img = eye_img.astype("float") / 255.0
        eye_img = np.expand_dims(eye_img, axis=0)
        eye_img = np.expand_dims(eye_img, axis=-1)
        return eye_img
    except: return None

def bandpass(signal, fs, low=0.8, high=3.0):
    """Bandpass filter with robust error handling"""
    try:
        # Validate sampling frequency
        if fs <= 0 or fs < 5:
            return signal
        
        nyquist = fs / 2.0
        
        # Check if cutoff frequencies are valid
        if low >= nyquist or high >= nyquist or low >= high:
            return signal
        
        # Normalized frequencies (must be 0 < Wn < 1)
        low_norm = np.clip(low / nyquist, 0.01, 0.99)
        high_norm = np.clip(high / nyquist, low_norm + 0.01, 0.99)
        
        # Design and apply filter
        b, a = butter(2, [low_norm, high_norm], btype='band')
        filtered = lfilter(b, a, signal)
        return filtered
    except Exception as e:
        print(f"Bandpass filter error: {e}")
        return signal

def normalize(val, min_v, max_v):
    return np.clip((val - min_v) / (max_v - min_v), 0, 1)

def get_rppg_roi(frame, shape, idx, size=15):
    x, y = shape[idx]
    x, y = max(0, x-size), max(0, y-size)
    w, h = size*2, size*2
    return frame[y:y+h, x:x+w]

def rppg_pos(roi):
    roi = roi.astype(np.float32) / 255.0
    r, g, b = np.mean(roi[:,:,0]), np.mean(roi[:,:,1]), np.mean(roi[:,:,2])
    X = np.array([3*r - 2*g, 1.5*r + g - 1.5*b])
    return X

def rppg_chrom(roi):
    roi = roi.astype(np.float32) / 255.0
    r, g, b = np.mean(roi[:,:,0]), np.mean(roi[:,:,1]), np.mean(roi[:,:,2])
    X = r - g
    Y = r + g - 2*b
    return X, Y

def speak_alert(engine, message):
    if engine is None: return
    def _speak():
        try:
            engine.say(message)
            engine.runAndWait()
        except: pass
    threading.Thread(target=_speak, daemon=True).start()

# --- THREADED CAMERA CLASS ---
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                with self.lock:
                    if ret:
                        self.ret = ret
                        self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    cnn, lstm = None, None
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        cnn_path = os.path.join(base_dir, "drowsiness_model.keras")
        if os.path.exists(cnn_path):
            cnn = keras.models.load_model(cnn_path)
            st.sidebar.success("✅ CNN Model loaded")
        else:
            st.sidebar.warning("⚠️ CNN model not found")
    except Exception as e:
        st.sidebar.error(f"❌ CNN load error: {e}")
    
    try:
        lstm_path = os.path.join(base_dir, "fatigue_lstm.keras")
        if os.path.exists(lstm_path):
            lstm = keras.models.load_model(lstm_path)
            st.sidebar.success("✅ LSTM Model loaded")
        else:
            st.sidebar.warning("⚠️ LSTM model not found")
    except Exception as e:
        st.sidebar.error(f"❌ LSTM load error: {e}")
    
    return cnn, lstm

@st.cache_resource
def init_audio():
    try:
        mixer.init()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load both audio files as Sound objects
        beep_sound = mixer.Sound(os.path.join(base_dir, "music.wav"))  # Beep for moderate warnings
        alert_sound = mixer.Sound(os.path.join(base_dir, "alert-369027.mp3"))  # Voice alert for critical
        
        # Set volumes
        beep_sound.set_volume(0.7)
        alert_sound.set_volume(1.0)
        
        return {'enabled': True, 'beep': beep_sound, 'alert': alert_sound}
    except:
        st.sidebar.warning("⚠️ Audio system not available")
        return {'enabled': False, 'beep': None, 'alert': None}

@st.cache_resource
def init_tts():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        return engine
    except:
        return None

# --- MAIN DETECTION LOGIC ---
if st.sidebar.checkbox("🚀 Start Monitoring", key='start_monitor'):
    
    # Load resources
    model, lstm_model = load_models()
    audio_system = init_audio()
    audio_enabled = audio_system['enabled']
    tts_engine = init_tts()
    
    # UI Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_frame = st.empty()
    
    with col2:
        st.subheader("📊 Live Metrics")
        st_alert = st.empty()
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st_ear = st.empty()
            st_mar = st.empty()
        with metric_col2:
            st_att = st.empty()
            st_hr = st.empty()
        
        st_status = st.empty()
        st.markdown("---")
        st_chart = st.empty()
        
        st.markdown("---")
        st_snapshot = st.empty()
    
    # Initialize MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,  # Higher for robustness
        min_tracking_confidence=0.6
    )
    
    hands_detector = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,  # Higher to reduce false positives
        min_tracking_confidence=0.7
    )
    
    # State variables
    MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
    MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    
    # Buffers with improved sizes
    ear_buffer = deque(maxlen=45)  # 1.5s at 30fps
    perclos_buffer = deque(maxlen=300)  # 10s window
    feature_buffer = deque(maxlen=30)
    attention_buffer = deque(maxlen=30)
    pos_buffer = deque(maxlen=300)
    chrom_buffer = deque(maxlen=300)
    time_buffer = deque(maxlen=300)
    chart_data = deque(maxlen=300)
    
    # Counters with hysteresis
    drowsy_counter = 0
    yawn_counter = 0
    distraction_counter = 0
    alert_cooldown = 0
    flag = 0
    prev_alert_level = 0
    
    # Session tracking
    session_start_time = time.time()
    alert_count = 0
    frame_count = 0
    
    # Voice alert tracking
    last_voice_alert_time = 0
    last_attention_level = 4
    VOICE_ALERT_COOLDOWN = 15
    last_beep_time = 0
    BEEP_INTERVAL = 2
    
    # Feature flags
    show_mesh = st.sidebar.checkbox("Show Face Mesh", value=False)
    mesh_style = st.sidebar.radio("Mesh Style", ["Contours", "Full Tesselation"], index=0)
    privacy_mode = st.sidebar.checkbox("Privacy Mode (No snapshots)", value=False)
    
    # Warmup notification
    if not st.session_state.calibrated:
        st.warning("⚠️ System not calibrated - using default thresholds. Click 'Calibrate Thresholds' for better accuracy.")
    
    # Start camera
    cap = ThreadedCamera(0)
    
    try:
        # FPS tracking
        fps = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        # Adaptive thresholds
        adaptive_thresh = 0.65
        fatigue_baseline = deque(maxlen=300)
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.error("❌ Camera error")
                break
            
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            h, w, c = frame.shape
            frame_count += 1
            
            # FPS calculation
            fps_counter += 1
            if fps_counter >= 30:
                fps = 30 / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # MediaPipe processing
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            hand_results = hands_detector.process(rgb)
            
            # Default values
            ear, mar, yaw, pitch = 0.3, 0.4, 0, 0
            perclos = 0
            lstm_fatigue = 0
            heart_rate = 70
            hrv = 0.03
            attention = 100
            ml_drowsy = False
            distraction_type = None
            is_phone = False
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])
                
                # Extract eye regions
                leftEye = shape[MP_LEFT_EYE]
                rightEye = shape[MP_RIGHT_EYE]
                
                # Calculate EAR
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                
                # EAR buffering
                ear_buffer.append(ear)
                perclos_buffer.append(1 if ear < EAR_THRESH else 0)
                
                # PERCLOS calculation
                perclos = sum(perclos_buffer) / max(len(perclos_buffer), 1)
                
                # EAR trend analysis
                slope = 0
                ear_std = 0
                if len(ear_buffer) >= 30:
                    recent_ear = list(ear_buffer)[-30:]
                    x = np.arange(30)
                    y = np.array(recent_ear)
                    slope = np.polyfit(x, y, 1)[0]
                    ear_std = np.std(recent_ear)
                
                # MAR (yawning)
                mar = mouth_aspect_ratio(shape)
                if mar > MAR_THRESH:
                    yawn_counter += 1
                else:
                    yawn_counter = max(0, yawn_counter - 2)
                yawning = yawn_counter > 20
                
                # Head pose
                yaw, pitch = get_head_pose(shape)
                
                # Phone/hand detection (improved)
                is_phone = False
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        # Check if hand is near face (phone usage indicator)
                        hand_coords = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark])
                        hand_center = np.mean(hand_coords, axis=0)
                        face_center = np.mean(shape, axis=0)
                        dist = np.linalg.norm(hand_center - face_center)
                        
                        if dist < w * 0.35:  # Hand within 35% of frame width from face
                            is_phone = True
                            break
                
                # Distraction logic (stricter)
                looking_sideways = abs(yaw) > YAW_WARN
                looking_up_down = abs(pitch) > PITCH_WARN
                
                if is_phone:
                    distraction_counter += 1
                elif looking_sideways or looking_up_down:
                    distraction_counter += 1
                else:
                    distraction_counter = max(0, distraction_counter - 3)  # Fast recovery
                
                # Determine distraction type (only after sustained detection)
                if distraction_counter > 45:  # 1.5 seconds
                    if is_phone:
                        if mar > 0.5:
                            distraction_type = "EATING/DRINKING"
                        else:
                            distraction_type = "PHONE USAGE"
                    elif abs(yaw) > YAW_CRITICAL:
                        distraction_type = f"CRITICAL TURN ({int(abs(yaw))}°)"
                    elif abs(yaw) > YAW_WARN:
                        distraction_type = f"LOOKING {'LEFT' if yaw < 0 else 'RIGHT'}"
                    elif abs(pitch) > PITCH_CRITICAL:
                        distraction_type = f"CRITICAL TILT ({int(abs(pitch))}°)"
                    elif abs(pitch) > PITCH_WARN:
                        distraction_type = f"LOOKING {'DOWN' if pitch > 0 else 'UP'}"
                
                # rPPG heart rate
                rois = [get_rppg_roi(frame, shape, 10), get_rppg_roi(frame, shape, 330), get_rppg_roi(frame, shape, 101)]
                pos_signals = [rppg_pos(roi) for roi in rois if roi.size > 0]
                chrom_signals = [rppg_chrom(roi) for roi in rois if roi.size > 0]
                
                if pos_signals:
                    pos_signal = np.mean(pos_signals, axis=0)
                    chrom_signal = np.mean(chrom_signals, axis=0)
                    pos_buffer.append(pos_signal)
                    chrom_buffer.append(chrom_signal)
                    time_buffer.append(time.time())
                
                if len(pos_buffer) > 120:
                    pos_sig = np.array(pos_buffer)
                    chrom_sig = np.array(chrom_buffer)
                    fused = pos_sig[:,0] - pos_sig[:,1] + chrom_sig[:,0] - chrom_sig[:,1]
                    
                    times = np.array(time_buffer)
                    fs = len(times) / (times[-1] - times[0])
                    filtered = bandpass(fused, fs)
                    peaks, _ = find_peaks(filtered, distance=max(1, int(fs*0.5)), prominence=0.1)
                    
                    if len(peaks) > 2:
                        rr_intervals = np.diff(times[peaks])
                        rr_intervals = rr_intervals[(rr_intervals > 0.5) & (rr_intervals < 1.5)]
                        if len(rr_intervals) > 0:
                            heart_rate = np.clip(60 / np.mean(rr_intervals), 45, 120)
                            hrv = np.std(rr_intervals)
                
                # ML drowsiness detection
                if model is not None:
                    l_input = preprocess_eye(gray, leftEye)
                    r_input = preprocess_eye(gray, rightEye)
                    
                    if l_input is not None and r_input is not None:
                        l_pred = model.predict(l_input, verbose=0)[0][0]
                        r_pred = model.predict(r_input, verbose=0)[0][0]
                        if l_pred < 0.35 and r_pred < 0.35:  # Both eyes closed with confidence
                            ml_drowsy = True
                
                # LSTM fatigue prediction
                if lstm_model is not None:
                    feature_vector = [
                        ear,
                        slope,
                        ear_std,
                        perclos,
                        mar,
                        abs(yaw) / 90.0,
                        abs(pitch) / 90.0,
                        int(ml_drowsy),
                        heart_rate / 100.0,
                        hrv
                    ]
                    feature_buffer.append(feature_vector)
                    
                    if len(feature_buffer) == 30:
                        seq = np.array(feature_buffer).reshape(1, 30, 10)
                        lstm_fatigue = lstm_model.predict(seq, verbose=0)[0][0]
                        
                        fatigue_baseline.append(lstm_fatigue)
                        if len(fatigue_baseline) > 100:
                            mean_base = np.mean(fatigue_baseline)
                            std_base = np.std(fatigue_baseline)
                            adaptive_thresh = mean_base + 3.0 * std_base
                
                # ==================== IMPROVED ATTENTION SCORE ====================
                attention = 100
                
                # PERCLOS (strongest drowsiness indicator)
                if perclos > PERCLOS_CRITICAL:
                    attention -= 40
                elif perclos > PERCLOS_WARN:
                    attention -= 20
                elif perclos > 0.10:
                    attention -= 10
                
                # LSTM fatigue
                if lstm_fatigue > adaptive_thresh:
                    attention -= 35
                elif lstm_fatigue > adaptive_thresh * 0.75:
                    attention -= 20
                
                # EAR threshold
                if ear < EAR_THRESH * 0.85:
                    attention -= 25
                elif ear < EAR_THRESH:
                    attention -= 12
                
                # Yawning
                if yawning:
                    attention -= 18
                
                # Distraction penalties
                if abs(yaw) > YAW_CRITICAL:
                    attention -= 40
                elif abs(yaw) > YAW_WARN:
                    attention -= 22
                
                if abs(pitch) > PITCH_CRITICAL:
                    attention -= 35
                elif abs(pitch) > PITCH_WARN:
                    attention -= 18
                
                # Phone usage
                if is_phone:
                    attention -= 30
                
                # Heart rate anomalies
                if heart_rate < 55:
                    attention -= 15
                elif heart_rate > 95:
                    attention -= 12
                
                if hrv < 0.02:
                    attention -= 12
                
                # Smooth attention
                attention = int(np.clip(attention, 0, 100))
                attention_buffer.append(attention)
                smooth_attention = int(np.mean(attention_buffer))
                
                # Warmup period (prevent false alarms)
                if frame_count < 120:
                    smooth_attention = max(smooth_attention, 65)
                
                attention = smooth_attention
                
                # ==================== ATTENTION-BASED VOICE ALERTS ====================
                current_time = time.time()
                current_attention_level = 0
                voice_message = None
                should_beep = False
                should_alert = False
                
                if attention >= 75:
                    current_attention_level = 4  # Fully alert
                elif 50 <= attention < 75:
                    current_attention_level = 3  # Mild concern
                    voice_message = "Be focused on the road."
                elif 40 <= attention < 50:
                    current_attention_level = 2  # Moderate - use beep
                    voice_message = "Be more focused. If you feel sleepy, please park the car outside the road and take rest."
                    should_beep = True
                elif attention < 40:
                    current_attention_level = 1  # Critical - use voice alert
                    should_alert = True
                
                if (current_attention_level < last_attention_level and 
                    current_time - last_voice_alert_time > VOICE_ALERT_COOLDOWN and
                    voice_message and frame_count > 120):
                    
                    speak_alert(tts_engine, voice_message)
                    last_voice_alert_time = current_time
                    logging.info(f"Voice alert: {voice_message} (Attention: {attention}%)")
                
                last_attention_level = current_attention_level
                
                # Audio alerts based on attention level
                if should_beep and current_time - last_beep_time > BEEP_INTERVAL:
                    # Moderate warning (40-50%): Play beep sound
                    if audio_enabled:
                        try:
                            if not mixer.get_busy():
                                audio_system['beep'].play()
                            last_beep_time = current_time
                        except: pass
                
                elif should_alert and current_time - last_beep_time > BEEP_INTERVAL:
                    # Critical alert (< 40%): Play voice alert sound
                    if audio_enabled:
                        try:
                            if not mixer.get_busy():
                                audio_system['alert'].play()
                            last_beep_time = current_time
                        except: pass
                
                # Stop sounds when attention improves
                if attention >= 50 and mixer.get_busy():
                    try:
                        mixer.stop()
                    except: pass

                # ==================== ALERT LOGIC ====================
                # Multi-factor confirmation required
                
                is_critically_drowsy = (
                    perclos > PERCLOS_CRITICAL and
                    ear < EAR_THRESH * 0.95 and
                    (lstm_model is None or lstm_fatigue > adaptive_thresh)
                )
                
                is_drowsy = (
                    (perclos > PERCLOS_WARN and ear < EAR_THRESH) or
                    (lstm_model and lstm_fatigue > adaptive_thresh * 0.8) or
                    (ml_drowsy and perclos > 0.12)
                )
                
                is_early_drowsy = (
                    perclos > 0.12 or
                    slope < -0.003 or
                    (ear < EAR_THRESH and frame_count > 120)
                )
                
                is_critically_distracted = (
                    (abs(yaw) > YAW_CRITICAL or abs(pitch) > PITCH_CRITICAL) and
                    distraction_counter > 60
                )
                
                is_distracted = (
                    distraction_type is not None and
                    distraction_counter > 45
                )
                
                # Determine alert level
                alert_triggered = False
                alert_level = 0
                alert_message = ""
                
                if is_critically_drowsy:
                    drowsy_counter += 1
                    if drowsy_counter > 30:  # 1 second sustained
                        alert_triggered = True
                        alert_level = 3
                        alert_message = "CRITICAL DROWSINESS"
                elif is_drowsy:
                    drowsy_counter += 1
                    if drowsy_counter > 20:
                        alert_triggered = True
                        alert_level = 2
                        alert_message = "DROWSINESS DETECTED"
                elif is_early_drowsy:
                    drowsy_counter = min(drowsy_counter + 1, 15)
                    if drowsy_counter > 12 and frame_count > 120:
                        alert_level = 1
                        alert_message = "EARLY DROWSINESS"
                else:
                    drowsy_counter = max(0, drowsy_counter - 2)
                
                # Distraction overrides
                if is_critically_distracted and alert_level < 3:
                    alert_triggered = True
                    alert_level = 3
                    alert_message = f"CRITICAL: {distraction_type}"
                elif is_distracted and alert_level < 2:
                    alert_triggered = True
                    alert_level = 2
                    alert_message = distraction_type
                
                # Cooldown to prevent spam
                if alert_cooldown > 0:
                    alert_cooldown -= 1
                    if alert_level < 3:
                        alert_triggered = False
                
                if alert_triggered:
                    flag += 1
                    if flag >= FRAME_CHECK and alert_cooldown == 0:
                        if flag == FRAME_CHECK and not privacy_mode:
                            alert_count += 1
                            st_snapshot.image(frame, channels="BGR", caption=f"Alert: {alert_message}")
                        
                        if alert_level == 3:
                            st_alert.error(f"🚨 CRITICAL: {alert_message}")
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
                            cv2.putText(frame, "CRITICAL ALERT", (10, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            # Audio is handled by attention-based system
                            alert_cooldown = 90  # 3 second cooldown
                        
                        elif alert_level == 2:
                            st_alert.warning(f"⚠️ WARNING: {alert_message}")
                            cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 10)
                            cv2.putText(frame, f"WARNING: {alert_message}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                            # Audio is handled by attention-based system
                            alert_cooldown = 60  # 2 second cooldown
                        
                        elif alert_level == 1:
                            st_alert.info(f"ℹ️ CAUTION: {alert_message}")
                            cv2.putText(frame, alert_message, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            alert_cooldown = 30  # 1 second cooldown
                        
                        logging.warning(f"Alert L{alert_level}: {alert_message}, Att: {attention}")
                else:
                    flag = max(0, flag - 1)
                    st_alert.empty()
                    # Audio is handled by attention-based system
                
                prev_alert_level = alert_level
                
                # ==================== VISUALIZATION ====================
                if show_mesh:
                    if "Full" in mesh_style:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                
                cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
                
                # Distraction indicator
                if distraction_type and distraction_counter > 30:
                    cv2.putText(frame, f"DISTRACTION: {distraction_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Status metrics
                color_state = (0, 255, 0) if attention > 70 else \
                             (0, 255, 255) if attention > 50 else \
                             (0, 165, 255) if attention > 30 else (0, 0, 255)
                color_state = (0, 255, 0) if attention >= 75 else \
                             (0, 255, 255) if attention >= 50 else \
                             (0, 165, 255) if attention >= 40 else (0, 0, 255)
                
                cv2.putText(frame, f"ATTENTION: {attention}%", (10, h-120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_state, 2)
                cv2.putText(frame, f"EAR: {ear:.3f} | PERCLOS: {perclos:.2%}", (10, h-85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Head: Yaw={int(yaw)}° Pitch={int(pitch)}°", (10, h-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"HR: {int(heart_rate)} BPM | HRV: {hrv:.3f}", (10, h-35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if lstm_model:
                    cv2.putText(frame, f"Fatigue: {lstm_fatigue:.2f}", (10, h-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255) if lstm_fatigue > adaptive_thresh else (0, 255, 0), 1)
                
                # Update UI metrics (throttled)
                if frame_count % 10 == 0:
                    st_ear.metric("EAR", f"{ear:.3f}")
                    st_mar.metric("MAR", f"{mar:.2f}")
                    st_att.metric("Attention", f"{attention}%")
                    st_hr.metric("HR / Fatigue", f"{int(heart_rate)} BPM / {lstm_fatigue:.2f}")
                    
                    status_text = "ALERT" if attention > 70 else \
                                 "CAUTION" if attention > 50 else \
                                 "WARNING" if attention > 30 else "DANGER"
                    status_text = "ALERT" if attention >= 75 else \
                                 "CAUTION" if attention >= 50 else \
                                 "WARNING" if attention >= 40 else "DANGER"
                    
                    status_color = "green" if attention > 70 else \
                                  "orange" if attention > 50 else "red"
                    status_color = "green" if attention >= 75 else \
                                  "orange" if attention >= 50 else "red"
                    
                    st_status.markdown(f"<h3 style='color:{status_color}; text-align: center;'>{status_text}</h3>",
                        unsafe_allow_html=True)
                    
                    chart_data.append(attention)
                    st_chart.line_chart(list(chart_data))
                
                cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            st_frame.image(frame, channels="BGR", width="stretch")
            
    finally:
        cap.stop()
        hands_detector.close()
        face_mesh.close()
        cv2.destroyAllWindows()
        
        # Session report
        st.markdown("---")
        st.success("✅ Monitoring Session Ended")
        duration = time.time() - session_start_time
        col1, col2, col3 = st.columns(3)
        col1.metric("Session Duration", f"{int(duration)}s")
        col2.metric("Total Alerts", alert_count)
        col3.metric("Avg Attention", f"{int(np.mean(chart_data)) if chart_data else 0}%")
else:
    st.info("📌 Check 'Start Monitoring' in the sidebar to begin.")
    st.markdown("""
    ### 🎯 Improved Features:
    - **Personalized Calibration**: Click 'Calibrate Thresholds' for your face
    - **Reduced False Alarms**: Multi-factor confirmation & temporal filtering
    - **Robust Distraction Detection**: Sustained head pose + hand tracking
    - **Adaptive Thresholds**: System learns your baseline over time
    - **Real-World Tested**: Optimized for actual driving conditions
    
    ### 📊 Detection Indicators:
    - **PERCLOS**: Industry-standard drowsiness metric (% eye closure)
    - **EAR Trend**: Early fatigue detection via declining eye openness
    - **Head Pose**: Multi-angle distraction detection (yaw & pitch)
    - **Heart Rate**: Physiological fatigue indicator (rPPG)
    - **ML Models**: CNN for eye state + LSTM for fatigue prediction
    """)