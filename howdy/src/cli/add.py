# Save the face of the user in encoded form

# Import required modules
import time
import os
import sys
import json
import configparser
import builtins
import numpy as np
import paths_factory
import onnxruntime as ort

class ArcFaceEncoder:
	"""Face encoder using ArcFace ONNX model."""
	
	def __init__(self, model_path):
		self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
		self.input_name = self.session.get_inputs()[0].name
		
	def preprocess(self, frame, landmarks):
		"""Align and crop face based on landmarks."""
		x, y, w, h = landmarks.rect.left(), landmarks.rect.top(), landmarks.rect.width(), landmarks.rect.height()
		face_img = frame[max(0, y):y+h, max(0, x):x+w]
		face_img = cv2.resize(face_img, (112, 112))
		
		# Normalize
		face_img = face_img.astype(np.float32)
		face_img = (face_img / 255.0 - 0.5) / 0.5
		face_img = np.transpose(face_img, (2, 0, 1))
		face_img = np.expand_dims(face_img, axis=0)
		return face_img

	def encode(self, frame, landmarks):
		"""Generate 512D embedding."""
		blob = self.preprocess(frame, landmarks)
		net_out = self.session.run(None, {self.input_name: blob})
		embeddings = net_out[0]
		
		# L2 Normalize
		norm = np.linalg.norm(embeddings)
		if norm > 1e-6:
			embeddings = embeddings / norm
			
		return embeddings.flatten()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recorders.video_capture import VideoCapture
from i18n import _

# Try to import dlib and give a nice error if we can't
# Add should be the first point where import issues show up
try:
	import dlib
except ImportError as err:
	print(err)

	print(_("\nCan't import the dlib module, check the output of"))
	print("pip3 show dlib")
	sys.exit(1)

# OpenCV needs to be imported after dlib
import cv2

# Test if at lest 1 of the data files is there and abort if it's not
if not os.path.isfile(paths_factory.shape_predictor_5_face_landmarks_path()):
	print(_("Data files have not been downloaded, please run the following commands:"))
	print("\n\tcd " + paths_factory.dlib_data_dir_path())
	print("\tsudo ./install.sh\n")
	sys.exit(1)

# Read config from disk
config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

use_cnn = config.getboolean("core", "use_cnn", fallback=False)
if use_cnn:
	face_detector = dlib.cnn_face_detection_model_v1(paths_factory.mmod_human_face_detector_path())
else:
	face_detector = dlib.get_frontal_face_detector()

pose_predictor = dlib.shape_predictor(paths_factory.shape_predictor_5_face_landmarks_path())

# Initialize ArcFace encoder instead of dlib ResNet
model_path = os.path.join(paths_factory.dlib_data_dir_path(), "arcface_buffalo_l.onnx")
if not os.path.isfile(model_path):
	print(_("ArcFace model not found at {}, please download it").format(model_path))
	sys.exit(1)
	
face_encoder = ArcFaceEncoder(model_path)

user = builtins.howdy_user
# The permanent file to store the encoded model in
enc_file = paths_factory.user_model_path(user)
# Known encodings
encodings = []

# Make the ./models folder if it doesn't already exist
if not os.path.exists(paths_factory.user_models_dir_path()):
	print(_("No face model folder found, creating one"))
	os.makedirs(paths_factory.user_models_dir_path())

# To try read a premade encodings file if it exists
try:
	encodings = json.load(open(enc_file))
except FileNotFoundError:
	encodings = []

# Print a warning if too many encodings are being added
if len(encodings) > 3:
	print(_("NOTICE: Each additional model slows down the face recognition engine slightly"))
	print(_("Press Ctrl+C to cancel\n"))

# Make clear what we are doing if not human
if not builtins.howdy_args.plain:
	print(_("Adding face model for the user ") + user)

# Set the default label
label = "Initial model"

# some id's can be skipped, but the last id is always the maximum
next_id = encodings[-1]["id"] + 1 if encodings else 0

# Get the label from the cli arguments if provided
if builtins.howdy_args.arguments:
	label = builtins.howdy_args.arguments[0]

# Or set the default label
else:
	label = _("Model #") + str(next_id)

# Keep de default name if we can't ask questions
if builtins.howdy_args.y:
	print(_('Using default label "%s" because of -y flag') % (label, ))
else:
	# Ask the user for a custom label
	label_in = input(_("Enter a label for this new model [{}]: ").format(label))

	# Set the custom label (if any) and limit it to 24 characters
	if label_in != "":
		label = label_in[:24]

# Remove illegal characters
if "," in label:
	print(_("NOTICE: Removing illegal character \",\" from model name"))
	label = label.replace(",", "")

# Prepare the metadata for insertion
insert_model = {
	"time": int(time.time()),
	"label": label,
	"id": next_id,
	"data": []
}

# Set up video_capture
video_capture = VideoCapture(config)

dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)
multi_angle = config.getboolean("video", "multi_angle_enrollment", fallback=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Define angles for multi-angle enrollment
angles = [
	("straight ahead", None),
	("slightly left", "left"),
	("slightly right", "right"),
	("slightly up", "up"),
	("slightly down", "down")
]

if multi_angle:
	print(_("\nMulti-angle enrollment enabled."))
	print(_("Please position your face at different angles as prompted."))
	time.sleep(2)
else:
	print(_("\nPlease look straight into the camera"))
	time.sleep(2)


def capture_face_encoding(video_capture, prompt_text, angle_hint=None):
	"""Capture a single face encoding with the given prompt."""
	global frames, valid_frames, dark_tries, dark_running_total
	
	print(_("\nPlease look {}").format(prompt_text))
	if angle_hint:
		print(_("Turn your head {}").format(angle_hint))
	time.sleep(1)
	
	frames = 0
	valid_frames = 0
	dark_tries = 0
	dark_running_total = 0
	face_locations = None
	
	while frames < 60:
		frames += 1
		frame, gsframe = video_capture.read_frame()
		gsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gsframe = clahe.apply(gsframe)
		
		hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
		hist_total = np.sum(hist)
		darkness = (hist[0] / hist_total * 100)
		
		if (hist_total == 0) or (darkness == 100):
			continue
		
		dark_running_total += darkness
		valid_frames += 1
		
		if (darkness > dark_threshold):
			dark_tries += 1
			continue
		
		face_locations = face_detector(gsframe, 1)
		
		if face_locations:
			break
	
	if not face_locations:
		if valid_frames == 0:
			print(_("Camera saw only black frames - is IR emitter working?"))
		elif valid_frames == dark_tries:
			print(_("All frames were too dark, please check dark_threshold in config"))
		else:
			print(_("No face detected for {} angle, skipping").format(prompt_text))
		return None
	
	if len(face_locations) > 1:
		print(_("Multiple faces detected for {} angle, skipping").format(prompt_text))
		return None
	
	face_location = face_locations[0]
	if use_cnn:
		face_location = face_location.rect
	
	face_landmark = pose_predictor(frame, face_location)
	face_encoding = face_encoder.encode(frame, face_landmark)
	
	print(_("Captured {} angle successfully").format(prompt_text))
	return face_encoding


if multi_angle:
	for prompt, hint in angles:
		encoding = capture_face_encoding(video_capture, prompt, hint)
		if encoding is not None:
			insert_model["data"].append(encoding.tolist())
	
	if len(insert_model["data"]) == 0:
		print(_("\nNo face encodings captured from any angle, aborting"))
		sys.exit(1)
	
	print(_("\nCaptured {} face encodings from different angles").format(len(insert_model["data"])))
else:
	enc = []
	frames = 0
	valid_frames = 0
	dark_tries = 0
	dark_running_total = 0
	face_locations = None
	
	while frames < 60:
		frames += 1
		frame, gsframe = video_capture.read_frame()
		gsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gsframe = clahe.apply(gsframe)
		
		hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
		hist_total = np.sum(hist)
		darkness = (hist[0] / hist_total * 100)
		
		if (hist_total == 0) or (darkness == 100):
			continue
		
		dark_running_total += darkness
		valid_frames += 1
		
		if (darkness > dark_threshold):
			dark_tries += 1
			continue
		
		face_locations = face_detector(gsframe, 1)
		
		if face_locations:
			break
	
	if not face_locations:
		if valid_frames == 0:
			print(_("Camera saw only black frames - is IR emitter working?"))
		elif valid_frames == dark_tries:
			print(_("All frames were too dark, please check dark_threshold in config"))
			print(_("Average darkness: {avg}, Threshold: {threshold}").format(avg=str(dark_running_total / valid_frames), threshold=str(dark_threshold)))
		else:
			print(_("No face detected, aborting"))
		sys.exit(1)
	
	if len(face_locations) > 1:
		print(_("Multiple faces detected, aborting"))
		sys.exit(1)
	
	face_location = face_locations[0]
	if use_cnn:
		face_location = face_location.rect
	
	face_landmark = pose_predictor(frame, face_location)
	face_encoding = face_encoder.encode(frame, face_landmark)
	
	insert_model["data"].append(face_encoding.tolist())

video_capture.release()

# Insert full object into the list
encodings.append(insert_model)

# Save the new encodings to disk
with open(enc_file, "w") as datafile:
	json.dump(encodings, datafile)

# Give let the user know how it went
print(_("""\nScan complete
Added a new model to """) + user)
