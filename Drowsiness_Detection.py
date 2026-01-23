import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")
from scipy.spatial import distance
from pygame import mixer
import imutils
import cv2
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from collections import deque
from scipy.signal import butter, lfilter, find_peaks
import math
import time
import sys
import threading

# Try to import TTS
try:
	import pyttsx3
	tts_engine = pyttsx3.init()
	tts_engine.setProperty('rate', 150)
	tts_engine.setProperty('volume', 0.9)
	TTS_AVAILABLE = True
except:
	TTS_AVAILABLE = False
	tts_engine = None
	print("[INFO] Text-to-speech not available. Install with: pip install pyttsx3")

if np.__version__.startswith("2"):
	print(f"\n[CRITICAL] Incompatible NumPy version detected: {np.__version__}")
	print("TensorFlow 2.10 requires NumPy 1.x. Run: pip install \"numpy<2.0\"")
	sys.exit(1)

try:
	import mediapipe as mp
except ImportError as e:
	print(f"\n[CRITICAL] MediaPipe import failed: {e}")
	print("FIX: Try upgrading libraries: pip install \"tensorflow==2.15.0\" \"mediapipe==0.10.9\" \"numpy<2.0\"")
	sys.exit(1)

try:
	import tensorflow as tf
	try:
		from tensorflow import keras
	except ImportError:
		import keras
except Exception as e:
	tf = None
	print(f"\n[ERROR] TensorFlow import failed: {e}")
	if "DLL" in str(e):
		print("FIX: Install 'Microsoft Visual C++ Redistributable' (x64) from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
	elif "symbol" in str(e) or "descriptor" in str(e) or "version" in str(e):
		print("FIX: Version mismatch. Try: pip install \"tensorflow==2.15.0\" \"mediapipe==0.10.9\" \"numpy<2.0\"")
	print("System running in Fallback Mode (EAR only).\n")

mixer.init()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load both audio files
beep_sound = mixer.Sound(os.path.join(BASE_DIR, "music.wav"))  # Beep for moderate warnings
alert_sound = mixer.Sound(os.path.join(BASE_DIR, "alert-369027.mp3"))  # Voice alert for critical situations

# Set volumes
beep_sound.set_volume(0.7)
alert_sound.set_volume(1.0)

# Load the pre-trained Keras model
model = None
lstm_model = None
model_path = os.path.join(BASE_DIR, "drowsiness_model.keras")
lstm_path = os.path.join(BASE_DIR, "fatigue_lstm.keras")
try:
	if tf is not None and os.path.exists(model_path):
		model = keras.models.load_model(model_path)
		print("CNN Model loaded successfully.")
	else:
		print("Warning: 'drowsiness_model.keras' not found.")

	if tf is not None and os.path.exists(lstm_path):
		lstm_model = keras.models.load_model(lstm_path)
		print("LSTM Model loaded successfully.")
	else:
		print("Warning: 'fatigue_lstm.keras' not found.")
except Exception as e:
	print(f"Error loading models: {e}.")

def preprocess_eye(frame, eye_points):
	try:
		(x, y, w, h) = cv2.boundingRect(eye_points)
		pad = 5
		x = max(0, x - pad)
		y = max(0, y - pad)
		w += 2 * pad
		h += 2 * pad
		eye_img = frame[y:y+h, x:x+w]
		if eye_img.size == 0:
			return None
		eye_img = cv2.resize(eye_img, (24, 24))
		eye_img = eye_img.astype("float") / 255.0
		eye_img = np.expand_dims(eye_img, axis=0)
		eye_img = np.expand_dims(eye_img, axis=-1)
		return eye_img
	except Exception:
		return None

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

def draw_timeline(frame, values, y_offset, color):
	h, w = frame.shape[:2]
	values = np.array(values)
	if len(values) < 2:
		return
	for i in range(1, len(values)):
		x1 = int((i-1) * w / 600)
		x2 = int(i * w / 600)
		y1 = int(y_offset - values[i-1] * 80)
		y2 = int(y_offset - values[i] * 80)
		cv2.line(frame, (x1, y1), (x2, y2), color, 2)

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

def mouth_aspect_ratio(shape):
	ver = distance.euclidean(shape[13], shape[14])
	hor = distance.euclidean(shape[78], shape[308])
	mar = ver / hor
	return mar

def get_head_pose(shape):
	"""Calculate head pose angles (yaw, pitch)"""
	nose = shape[1]
	left_eye = shape[33]
	right_eye = shape[263]
	chin = shape[152]
	
	# Yaw (left-right turn)
	eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
	nose_to_center = nose[0] - eye_center[0]
	eye_width = abs(right_eye[0] - left_eye[0])
	yaw = (nose_to_center / eye_width) * 90 if eye_width > 0 else 0
	
	# Pitch (up-down tilt)
	dy = chin[1] - nose[1]
	dx = chin[0] - nose[0]
	pitch = math.degrees(math.atan2(dy, dx)) - 90
	
	return yaw, pitch

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# ==================== IMPROVED CONFIGURATION ====================
# These values are more robust for real-world scenarios

# EAR Thresholds - Personalized & Adaptive
BASE_EAR_THRESH = 0.21  # Lower baseline for more tolerance
CALIBRATION_FRAMES = 90  # 3 seconds at 30fps for calibration

# Temporal Requirements - Prevent instant false alarms
MIN_DROWSY_FRAMES = 25  # ~0.8s continuous drowsiness required
MIN_YAWN_FRAMES = 20  # ~0.7s continuous yawn
MIN_DISTRACTION_FRAMES = 45  # ~1.5s continuous distraction

# PERCLOS - Industry standard
PERCLOS_WINDOW = 300  # 10 seconds window
PERCLOS_CRITICAL = 0.35  # 35% closure in 10s = drowsy

# MAR Threshold - More conservative
MAR_THRESH = 0.65  # Higher to avoid false yawn detection

# Head Pose - More tolerant ranges
YAW_WARN_THRESH = 40  # degrees (looking sideways)
YAW_CRITICAL_THRESH = 55  # degrees
PITCH_WARN_THRESH = 35  # degrees (looking up/down)
PITCH_CRITICAL_THRESH = 50  # degrees

# Heart Rate - Normal driving range
HR_MIN_NORMAL = 55  # Below = too relaxed
HR_MAX_NORMAL = 95  # Above = stressed
HRV_MIN_NORMAL = 0.02  # Below = fatigued

# Attention Score - Multi-tier system
ATTENTION_ALERT = 75  # Above = fully alert
ATTENTION_CAUTION = 55  # Below = mild concern
ATTENTION_WARNING = 35  # Below = significant risk
ATTENTION_CRITICAL = 20  # Below = immediate danger

# Buffers
EAR_HISTORY = 45  # 1.5 seconds for trend analysis
SEQUENCE_LEN = 30
BASELINE_WINDOW = 300
PPG_WINDOW = 300

# Initialize buffers
ear_buffer = deque(maxlen=EAR_HISTORY)
perclos_buffer = deque(maxlen=PERCLOS_WINDOW)
feature_buffer = deque(maxlen=SEQUENCE_LEN)
fatigue_baseline = deque(maxlen=BASELINE_WINDOW)
pos_buffer = deque(maxlen=PPG_WINDOW)
chrom_buffer = deque(maxlen=PPG_WINDOW)
time_buffer = deque(maxlen=PPG_WINDOW)
attention_buffer = deque(maxlen=30)  # Smooth attention over 1 second

# Counters with hysteresis
drowsy_counter = 0
yawn_counter = 0
distraction_counter = 0
alert_cooldown = 0  # Prevent alert spam

# Calibration state
calibration_mode = True
calibrated_ear_baseline = 0.25
calibrated_mar_baseline = 0.4
calibration_counter = 0

# Online Learning
online_X = deque(maxlen=200)
online_y = deque(maxlen=200)
ONLINE_UPDATE_INTERVAL = 600  # Every 20 seconds
online_counter = 0

# Timeline
timeline_len = 600
fatigue_timeline = deque(maxlen=timeline_len)
attention_timeline = deque(maxlen=timeline_len)

# MediaPipe Setup
try:
	mp_face_mesh = mp.solutions.face_mesh
	face_mesh = mp_face_mesh.FaceMesh(
		max_num_faces=1, 
		refine_landmarks=True, 
		min_detection_confidence=0.5, 
		min_tracking_confidence=0.5
	)
except AttributeError as e:
	print(f"\n[CRITICAL ERROR] MediaPipe failed to load 'solutions'. Error: {e}")
	print("1. Ensure you installed 'Microsoft Visual C++ Redistributable' (linked above).")
	print("2. Check if you have a file named 'mediapipe.py' in this folder (rename it if yes).")
	try:
		import google.protobuf
		print(f"3. Current Protobuf version: {google.protobuf.__version__}")
		print("   Ensure Protobuf is compatible with your TensorFlow and MediaPipe versions.")
	except: pass
	sys.exit(1)
except Exception as e:
	print(f"\n[CRITICAL ERROR] MediaPipe init failed: {e}")
	sys.exit(1)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# MP Indices
MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def pseudo_label(lstm_fatigue, attention):
	"""Generate pseudo-labels for online learning - more conservative"""
	if attention > 85 and lstm_fatigue < 0.15:
		return 0  # clearly alert
	elif attention < 25 and lstm_fatigue > 0.85:
		return 1  # clearly fatigued
	else:
		return None  # uncertain, don't use for training

def speak_alert(message):
	"""Thread-safe voice alert"""
	if not TTS_AVAILABLE or tts_engine is None:
		return
	try:
		def _speak():
			try:
				tts_engine.say(message)
				tts_engine.runAndWait()
			except:
				pass
		threading.Thread(target=_speak, daemon=True).start()
	except Exception as e:
		print(f"[TTS Error] {e}")

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	flag = 0
	frame_count = 0
	
	# Voice alert tracking
	last_voice_alert_time = 0
	last_attention_level = 4  # Start at highest (>75)
	VOICE_ALERT_COOLDOWN = 15  # 15 seconds between voice alerts
	last_beep_time = 0
	BEEP_INTERVAL = 2  # 2 seconds between emergency beeps
	
	print("\n" + "="*60)
	print("IMPROVED DROWSINESS DETECTION SYSTEM")
	print("="*60)
	print("\nCALIBRATION MODE: Active for first 3 seconds")
	print("Please sit normally and look at the camera.")
	print("Keep your eyes open and mouth closed.\n")
	print("="*60 + "\n")
	
	while True:
		ret, frame = cap.read()
		if not ret: 
			break
		
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = clahe.apply(gray)
		frame_count += 1

		# MediaPipe Processing
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		results = face_mesh.process(rgb)
		
		if results.multi_face_landmarks:
			face_landmarks = results.multi_face_landmarks[0]
			h, w, c = frame.shape
			shape = np.array([[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark])
			
			leftEye = shape[MP_LEFT_EYE]
			rightEye = shape[MP_RIGHT_EYE]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			
			# ==================== CALIBRATION PHASE ====================
			if calibration_mode:
				ear_buffer.append(ear)
				calibration_counter += 1
				
				cv2.putText(frame, "CALIBRATING... Keep eyes open!", (10, 30), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
				cv2.putText(frame, f"Progress: {calibration_counter}/{CALIBRATION_FRAMES}", (10, 60),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
				
				if calibration_counter >= CALIBRATION_FRAMES:
					# Calculate personalized baseline
					calibrated_ear_baseline = np.mean(ear_buffer) - 0.05  # 5% below resting
					calibration_mode = False
					ear_buffer.clear()
					print(f"\n✓ CALIBRATION COMPLETE")
					print(f"  Your EAR Baseline: {calibrated_ear_baseline:.3f}")
					print(f"  Starting monitoring...\n")
				
				cv2.imshow("Frame", frame)
				key = cv2.waitKey(1) & 0xFF
				if key == ord("q"):
					break
				continue
			
			# Use calibrated threshold
			thresh = calibrated_ear_baseline
			
			# ==================== FEATURE EXTRACTION ====================
			ear_buffer.append(ear)
			perclos_buffer.append(1 if ear < thresh else 0)

			# EAR Trend Analysis (more robust)
			slope = 0
			ear_variance = 0
			if len(ear_buffer) >= 30:
				recent_ear = list(ear_buffer)[-30:]
				x = np.arange(30)
				y = np.array(recent_ear)
				slope = np.polyfit(x, y, 1)[0]
				ear_variance = np.var(recent_ear)

			# PERCLOS (Industry Standard)
			perclos = sum(perclos_buffer) / len(perclos_buffer) if len(perclos_buffer) > 0 else 0

			# EAR Variability (detect micro-sleep)
			ear_std = np.std(ear_buffer) if len(ear_buffer) > 20 else 0

			# YAWNING (with hysteresis)
			mar = mouth_aspect_ratio(shape)
			if mar > MAR_THRESH:
				yawn_counter += 1
			else:
				yawn_counter = max(0, yawn_counter - 2)  # Decay faster
			yawning = yawn_counter > MIN_YAWN_FRAMES

			# HEAD POSE (yaw, pitch)
			yaw, pitch = get_head_pose(shape)
			
			# DISTRACTION DETECTION (stricter temporal requirement)
			looking_away = abs(yaw) > YAW_WARN_THRESH or abs(pitch) > PITCH_WARN_THRESH
			if looking_away:
				distraction_counter += 1
			else:
				distraction_counter = max(0, distraction_counter - 3)  # Fast recovery
			
			distracted = distraction_counter > MIN_DISTRACTION_FRAMES

			# ==================== rPPG HEART RATE ====================
			rois = [get_rppg_roi(frame, shape, 10), get_rppg_roi(frame, shape, 330), get_rppg_roi(frame, shape, 101)]
			pos_signals = [rppg_pos(roi) for roi in rois if roi.size > 0]
			chrom_signals = [rppg_chrom(roi) for roi in rois if roi.size > 0]

			if pos_signals:
				pos_signal = np.mean(pos_signals, axis=0)
				chrom_signal = np.mean(chrom_signals, axis=0)
				pos_buffer.append(pos_signal)
				chrom_buffer.append(chrom_signal)
				time_buffer.append(time.time())

			heart_rate = 70  # Default
			hrv = 0.03
			if len(pos_buffer) > 120:
				pos_sig = np.array(pos_buffer)
				chrom_sig = np.array(chrom_buffer)
				fused = pos_sig[:,0] - pos_sig[:,1] + chrom_sig[:,0] - chrom_sig[:,1]
				
				times = np.array(time_buffer)
				time_span = times[-1] - times[0]
				
				# Validate time span (need at least 2 seconds)
				if time_span > 2.0:
					fs = len(times) / time_span
					
					# Validate fs (should be 5-100 Hz for video)
					if 5 <= fs <= 100:
						filtered = bandpass(fused, fs)
						peaks, _ = find_peaks(filtered, distance=fs*0.5, prominence=0.1)
						
						if len(peaks) > 2:
							rr_intervals = np.diff(times[peaks])
							rr_intervals = rr_intervals[(rr_intervals > 0.5) & (rr_intervals < 1.5)]  # Filter outliers
							if len(rr_intervals) > 0:
								heart_rate = 60 / np.mean(rr_intervals)
								hrv = np.std(rr_intervals)
								# Clamp to reasonable range
								heart_rate = np.clip(heart_rate, 45, 120)

			# ==================== ML PREDICTION ====================
			ml_drowsy = False
			if model is not None:
				l_input = preprocess_eye(gray, leftEye)
				r_input = preprocess_eye(gray, rightEye)
				
				if l_input is not None and r_input is not None:
					l_pred = model.predict(l_input, verbose=0)[0][0]
					r_pred = model.predict(r_input, verbose=0)[0][0]
					# Both eyes must be detected as closed
					if l_pred < 0.4 and r_pred < 0.4:
						ml_drowsy = True

			# ==================== LSTM FATIGUE PREDICTION ====================
			lstm_fatigue = 0
			adaptive_thresh = 0.65  # More conservative default
			
			feature_vector = [
				ear,
				slope,
				ear_std,
				perclos,
				mar,
				abs(yaw) / 90.0,  # Normalize angles
				abs(pitch) / 90.0,
				int(ml_drowsy),
				heart_rate / 100.0,
				hrv
			]
			feature_buffer.append(feature_vector)

			if lstm_model is not None and len(feature_buffer) == SEQUENCE_LEN:
				seq = np.array(feature_buffer).reshape(1, SEQUENCE_LEN, 10)
				lstm_fatigue = lstm_model.predict(seq, verbose=0)[0][0]

				# Adaptive Threshold with smoothing
				if len(fatigue_baseline) < BASELINE_WINDOW:
					fatigue_baseline.append(lstm_fatigue)
				
				if len(fatigue_baseline) > 100:
					mean_base = np.mean(fatigue_baseline)
					std_base = np.std(fatigue_baseline)
					adaptive_thresh = mean_base + 3.0 * std_base  # 3 sigma rule

			# ==================== ATTENTION SCORE (IMPROVED) ====================
			attention = 100
			
			# PERCLOS penalty (strongest indicator)
			if perclos > PERCLOS_CRITICAL:
				attention -= 35
			elif perclos > 0.25:
				attention -= 20
			elif perclos > 0.15:
				attention -= 10
			
			# Drowsiness penalties (with thresholds)
			if lstm_fatigue > adaptive_thresh:
				attention -= 30
			elif lstm_fatigue > adaptive_thresh * 0.75:
				attention -= 15
			
			if ear < thresh * 0.9:  # Very low EAR
				attention -= 20
			elif ear < thresh:
				attention -= 10
			
			# Yawning penalty
			if yawning:
				attention -= 15
			
			# Distraction penalties
			if abs(yaw) > YAW_CRITICAL_THRESH:
				attention -= 35
			elif abs(yaw) > YAW_WARN_THRESH:
				attention -= 20
			
			if abs(pitch) > PITCH_CRITICAL_THRESH:
				attention -= 30
			elif abs(pitch) > PITCH_WARN_THRESH:
				attention -= 15
			
			# Heart rate penalties (outside normal range)
			if heart_rate < HR_MIN_NORMAL:
				attention -= 15
			elif heart_rate > HR_MAX_NORMAL:
				attention -= 10
			
			if hrv < HRV_MIN_NORMAL:
				attention -= 10
			
			# Smooth attention score
			attention = int(np.clip(attention, 0, 100))
			attention_buffer.append(attention)
			smooth_attention = int(np.mean(attention_buffer))
			
			# Warmup period (prevent false alarms at start)
			if frame_count < 150:  # First 5 seconds
				smooth_attention = max(smooth_attention, 60)
			
			attention = smooth_attention

			# ==================== ATTENTION-BASED VOICE ALERTS ====================
			current_time = time.time()
			current_attention_level = 0
			voice_message = None
			should_beep = False
			should_alert = False
			
			# Determine attention level and message
			if attention >= 75:
				current_attention_level = 4  # Fully alert
			elif 50 <= attention < 75:
				current_attention_level = 3  # Mild concern
				voice_message = "Be focused on the road."
			elif 40 <= attention < 50:
				current_attention_level = 2  # Moderate concern - use beep
				voice_message = "Be more focused. If you feel sleepy, please park the car outside the road and take rest."
				should_beep = True
			elif attention < 40:
				current_attention_level = 1  # Critical - use voice alert
				should_alert = True
			
			# Voice alert logic (only if level decreased and cooldown passed)
			if (current_attention_level < last_attention_level and 
				current_time - last_voice_alert_time > VOICE_ALERT_COOLDOWN and
				voice_message and frame_count > 120):
				
				speak_alert(voice_message)
				last_voice_alert_time = current_time
				print(f"[VOICE ALERT] {voice_message} (Attention: {attention}%)")
			
			# Update last attention level
			last_attention_level = current_attention_level
			
			# Audio alerts based on attention level
			if should_beep and current_time - last_beep_time > BEEP_INTERVAL:
				# Moderate warning (40-50%): Play beep sound
				try:
					if not mixer.get_busy():
						beep_sound.play()
					last_beep_time = current_time
				except:
					pass
			
			elif should_alert and current_time - last_beep_time > BEEP_INTERVAL:
				# Critical alert (< 40%): Play voice alert sound
				try:
					if not mixer.get_busy():
						alert_sound.play()
					last_beep_time = current_time
				except:
					pass
			
			# Stop sounds when attention improves
			if attention >= 50 and mixer.get_busy():
				try:
					mixer.stop()
				except:
					pass

			# ==================== ALERT LOGIC (MULTI-TIER) ====================
			# Combine multiple indicators with AND/OR logic
			
			# Critical drowsiness requires multiple confirmations
			is_critically_drowsy = (
				perclos > PERCLOS_CRITICAL and 
				ear < thresh * 0.95 and
				(lstm_model is None or lstm_fatigue > adaptive_thresh)
			)
			
			# Warning level drowsiness
			is_drowsy = (
				(perclos > 0.25 and ear < thresh) or
				(lstm_model and lstm_fatigue > adaptive_thresh * 0.8) or
				(ml_drowsy and perclos > 0.15)
			)
			
			# Caution level
			is_early_drowsy = (
				perclos > 0.15 or
				slope < -0.003 or
				ear < thresh
			)
			
			# Distraction detection
			is_critically_distracted = (
				(abs(yaw) > YAW_CRITICAL_THRESH or abs(pitch) > PITCH_CRITICAL_THRESH) and
				distraction_counter > MIN_DISTRACTION_FRAMES
			)
			
			is_distracted = (
				(abs(yaw) > YAW_WARN_THRESH or abs(pitch) > PITCH_WARN_THRESH) and
				distraction_counter > MIN_DISTRACTION_FRAMES // 2
			)
			
			# Determine alert level
			alert_triggered = False
			alert_level = 0
			alert_message = ""
			
			if is_critically_drowsy:
				drowsy_counter += 1
				if drowsy_counter > MIN_DROWSY_FRAMES:
					alert_triggered = True
					alert_level = 3
					alert_message = "CRITICAL DROWSINESS"
			elif is_drowsy:
				drowsy_counter += 1
				if drowsy_counter > MIN_DROWSY_FRAMES // 2:
					alert_triggered = True
					alert_level = 2
					alert_message = "DROWSINESS DETECTED"
			elif is_early_drowsy:
				drowsy_counter = min(drowsy_counter + 1, MIN_DROWSY_FRAMES // 3)
				if drowsy_counter > MIN_DROWSY_FRAMES // 3:
					alert_level = 1
					alert_message = "EARLY DROWSINESS"
			else:
				drowsy_counter = max(0, drowsy_counter - 2)  # Fast decay
			
			# Distraction overrides (if more severe)
			if is_critically_distracted and alert_level < 3:
				alert_triggered = True
				alert_level = 3
				alert_message = f"CRITICAL DISTRACTION (Head: {int(max(abs(yaw), abs(pitch)))}°)"
			elif is_distracted and alert_level < 2:
				alert_triggered = True
				alert_level = 2
				if abs(yaw) > abs(pitch):
					alert_message = f"LOOKING SIDEWAYS ({int(abs(yaw))}°)"
				else:
					alert_message = f"LOOKING {('UP' if pitch < 0 else 'DOWN')} ({int(abs(pitch))}°)"
			
			# Cooldown logic to prevent alert spam
			if alert_cooldown > 0:
				alert_cooldown -= 1
				if alert_level < 3:  # Allow critical alerts through
					alert_triggered = False
			
			if alert_triggered:
				flag += 1
				if flag >= MIN_DROWSY_FRAMES and alert_cooldown == 0:
					# Trigger alarm
					cv2.putText(frame, f"*** {alert_message} ***", (10, 30), 
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					
					if alert_level >= 3:
						cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 15)
						# Emergency beep is handled by attention-based system above
						alert_cooldown = 60  # 2 second cooldown
					elif alert_level == 2:
						cv2.rectangle(frame, (0, 0), (w, h), (0, 165, 255), 10)
						# Audio is handled by attention system
						alert_cooldown = 30  # 1 second cooldown
			else:
				flag = max(0, flag - 1)
				# Don't stop beep here - let attention system handle it

			# ==================== VISUALIZATION ====================
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			
			# Status display
			color_state = (0, 255, 0) if attention > ATTENTION_ALERT else \
						  (0, 255, 255) if attention > ATTENTION_CAUTION else \
						  (0, 165, 255) if attention > ATTENTION_WARNING else (0, 0, 255)
			
			state_text = "ALERT" if attention > ATTENTION_ALERT else \
						 "CAUTION" if attention > ATTENTION_CAUTION else \
						 "WARNING" if attention > ATTENTION_WARNING else "DANGER"
			
			# Key metrics
			cv2.putText(frame, f"EAR: {ear:.3f} (Thresh: {thresh:.3f})", (10, h-120), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			cv2.putText(frame, f"PERCLOS: {perclos:.2%}", (10, h-95), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
			cv2.putText(frame, f"Head: Yaw={int(yaw)}° Pitch={int(pitch)}°", (10, h-70),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			cv2.putText(frame, f"HR: {int(heart_rate)} BPM | HRV: {hrv:.3f}", (10, h-45),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
			
			if lstm_model:
				cv2.putText(frame, f"LSTM Fatigue: {lstm_fatigue:.2f}", (10, h-20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
					(0, 0, 255) if lstm_fatigue > adaptive_thresh else (0, 255, 0), 1)
			
			# Attention score (large)
			cv2.putText(frame, f"ATTENTION: {attention}%", (10, 90), 
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_state, 2)
			cv2.putText(frame, f"STATE: {state_text}", (10, 130), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_state, 2)
			
			# Timeline graphs
			fatigue_timeline.append(lstm_fatigue)
			attention_timeline.append(attention / 100.0)
			draw_timeline(frame, fatigue_timeline, h-180, (0, 0, 255))
			draw_timeline(frame, attention_timeline, h-160, (0, 255, 0))
			
			# ==================== ONLINE LEARNING ====================
			if lstm_model is not None and len(feature_buffer) == SEQUENCE_LEN:
				label = pseudo_label(lstm_fatigue, attention)
				if label is not None:
					online_X.append(np.array(feature_buffer))
					online_y.append(label)
				
				online_counter += 1
				if online_counter % ONLINE_UPDATE_INTERVAL == 0 and len(online_X) > 50:
					X_train = np.array(online_X)
					y_train = np.array(online_y)
					lstm_model.compile(optimizer=keras.optimizers.Adam(5e-6), loss="binary_crossentropy")
					lstm_model.fit(X_train, y_train, epochs=1, batch_size=8, verbose=0)
					print(f"[{frame_count}] Online LSTM fine-tuned ({len(online_X)} samples)")

		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		elif key == ord("r"):  # Reset calibration
			calibration_mode = True
			calibration_counter = 0
			ear_buffer.clear()
			print("\n[INFO] Recalibrating...")
			
	cv2.destroyAllWindows()
	cap.release()