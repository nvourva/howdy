# Compare incoming video with known faces
# Running in a local python instance to get around PATH issues

# Debug settings
DEBUG = True
import logging
logger = logging.getLogger(__name__)

# Import time so we can start timing asap
import time

# Start timing
timings = {
	"st": time.time()
}

# Import required modules
import sys
import os
import json
import configparser
import dlib
import cv2
from datetime import timezone, datetime
import atexit
import subprocess
import snapshot
import numpy as np
import _thread as thread
import paths_factory
from recorders.video_capture import VideoCapture
from i18n import _

def exit(code=None):
	"""Exit while closing howdy-gtk properly"""
	global gtk_proc

	# Exit the auth ui process if there is one
	if "gtk_proc" in globals():
		gtk_proc.terminate()

	# Exit compare
	if code is not None:
		sys.exit(code)


def eye_aspect_ratio(eye):
	"""Compute the eye aspect ratio for blink detection."""
	vertical_1 = np.linalg.norm(eye[1] - eye[5])
	vertical_2 = np.linalg.norm(eye[2] - eye[4])
	horizontal = np.linalg.norm(eye[0] - eye[3])
	return (vertical_1 + vertical_2) / (2.0 * horizontal)


class BlinkDetector:
	"""Improved blink detector with proper state machine."""
	
	STATE_OPEN = 0
	STATE_CLOSING = 1
	STATE_CLOSED = 2
	STATE_OPENING = 3
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.blink_count = 0
		self.baseline_ear = None
		self.baseline_samples = []
		self.baseline_ready = False
		self.eye_closed = False
		self.closed_frames = 0
		self.last_blink_time = 0
	
	def detect(self, shape, ear_threshold, blink_frames, require_both_eyes, 
			   check_eye_recovery, min_blink_interval, ear_smoothing_window):
		"""Blink detection based on percentage drop from baseline EAR."""
		current_time = time.time()
		
		left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
		right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])
		
		left_ear = eye_aspect_ratio(left_eye)
		right_ear = eye_aspect_ratio(right_eye)
		
		if require_both_eyes:
			ear = min(left_ear, right_ear)
		else:
			ear = (left_ear + right_ear) / 2.0
		
		# Build baseline from frames with open eyes (EAR > 0.22 to exclude blinks)
		if len(self.baseline_samples) < 5:
			if ear > 0.22:  # Only use "open eye" frames for baseline
				self.baseline_samples.append(ear)
				debug_log.write(f"[{datetime.now()}] Baseline sample {len(self.baseline_samples)}/5: EAR={ear:.3f}\n")
				debug_log.flush()
			return self.blink_count, ear
		
		if not self.baseline_ready:
			self.baseline_ear = np.mean(self.baseline_samples)
			self.baseline_ready = True
			debug_log.write(f"[{datetime.now()}] Baseline EAR established: {self.baseline_ear:.3f}, threshold: {self.baseline_ear * ear_threshold:.3f}\n")
			debug_log.flush()
		
		# Blink threshold is percentage below baseline (ear_threshold is now a ratio, e.g., 0.7 = 70% of baseline)
		# If baseline is very low, we use a fixed minimum threshold to avoid impossible targets
		# Increased minimum threshold to 0.18 to be more sensitive
		blink_threshold = max(0.18, self.baseline_ear * ear_threshold)
		
		# State machine: detect close -> open cycle
		if ear < blink_threshold:
			self.closed_frames += 1
			if self.closed_frames >= 1 and not self.eye_closed:
				self.eye_closed = True
				debug_log.write(f"[{datetime.now()}] Eye closed detected (EAR: {ear:.3f})\n")
				debug_log.flush()
		else:
			# Eyes reopened
			if self.eye_closed:
				if current_time - self.last_blink_time >= min_blink_interval:
					self.blink_count += 1
					self.last_blink_time = current_time
					debug_log.write(f"[{datetime.now()}] Blink counted! Total: {self.blink_count} (EAR: {ear:.3f})\n")
					debug_log.flush()
				else:
					debug_log.write(f"[{datetime.now()}] Blink ignored (too soon: {current_time - self.last_blink_time:.2f}s)\n")
					debug_log.flush()
			
			# If EAR is high enough, we can reset the state even if we didn't count a blink
			# This helps if the baseline was bad or if the user's eyes are just naturally wide open
			if ear > self.baseline_ear * 0.8:
				self.eye_closed = False
				self.closed_frames = 0
		
		return self.blink_count, ear


blink_detector = BlinkDetector()


def get_head_pose(shape, frame_width, frame_height):
	"""Detect head pose direction using facial landmarks."""
	nose = np.array([shape.part(30).x, shape.part(30).y])
	left_eye_center = np.array([(shape.part(36).x + shape.part(39).x) / 2,
							   (shape.part(36).y + shape.part(39).y) / 2])
	right_eye_center = np.array([(shape.part(42).x + shape.part(45).x) / 2,
								 (shape.part(42).y + shape.part(45).y) / 2])
	
	eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
	eye_center = (left_eye_center + right_eye_center) / 2
	
	horizontal_offset = (nose[0] - eye_center[0]) / eye_distance
	vertical_offset = (nose[1] - eye_center[1]) / eye_distance
	
	mouth_center = np.array([(shape.part(62).x + shape.part(66).x) / 2,
							(shape.part(62).y + shape.part(66).y) / 2])
	nose_to_mouth = nose[1] - mouth_center[1]
	
	return {
		"horizontal": horizontal_offset,
		"vertical": vertical_offset,
		"nose_to_mouth": nose_to_mouth / eye_distance
	}


def detect_smile(shape):
	"""Detect smile using mouth landmarks."""
	mouth_left = np.array([shape.part(60).x, shape.part(60).y])
	mouth_right = np.array([shape.part(64).x, shape.part(64).y])
	mouth_top = np.array([shape.part(62).x, shape.part(62).y])
	mouth_bottom = np.array([shape.part(66).x, shape.part(66).y])
	
	mouth_width = np.linalg.norm(mouth_right - mouth_left)
	mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
	
	smile_ratio = mouth_width / (mouth_height + 0.1)
	
	return smile_ratio > 3.5


def get_challenge_direction(pose):
	"""Determine which direction the user is looking based on head pose."""
	h = pose["horizontal"]
	v = pose["vertical"]
	
	if h < -0.3:
		return "left"
	elif h > 0.3:
		return "right"
	elif v < -0.25:
		return "up"
	elif v > 0.25:
		return "down"
	return "center"


def detect_ir_camera(video_capture):
	"""Attempt to detect if camera is IR-capable."""
	try:
		frame, gsframe = video_capture.read_frame()
		if frame is None:
			return False, None
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
		
		mean_brightness = np.mean(gray)
		std_brightness = np.std(gray)
		
		is_likely_ir = mean_brightness < 80 or std_brightness < 30
		
		return is_likely_ir, {
			"mean_brightness": mean_brightness,
			"std_brightness": std_brightness,
			"is_likely_ir": is_likely_ir
		}
	except Exception:
		return False, None


def init_detector(lock):
	"""Start face detector, encoder and predictor in a new thread"""
	global face_detector, pose_predictor, pose_predictor_68, face_encoder

	# Test if at lest 1 of the data files is there and abort if it's not
	if not os.path.isfile(paths_factory.shape_predictor_5_face_landmarks_path()):
		print(_("Data files have not been downloaded, please run the following commands:"))
		print("\n\tcd " + paths_factory.dlib_data_dir_path())
		print("\tsudo ./install.sh\n")
		lock.release()
		exit(1)

	# Use the CNN detector if enabled
	if use_cnn:
		face_detector = dlib.cnn_face_detection_model_v1(paths_factory.mmod_human_face_detector_path())
	else:
		face_detector = dlib.get_frontal_face_detector()

	# Start the others regardless
	pose_predictor = dlib.shape_predictor(paths_factory.shape_predictor_5_face_landmarks_path())
	face_encoder = dlib.face_recognition_model_v1(paths_factory.dlib_face_recognition_resnet_model_v1_path())

	# Load 68-point predictor for liveness detection if available
	predictor_68_path = paths_factory.shape_predictor_68_face_landmarks_path()
	if os.path.isfile(predictor_68_path):
		pose_predictor_68 = dlib.shape_predictor(predictor_68_path)

	# Note the time it took to initialize detectors
	timings["ll"] = time.time() - timings["ll"]
	lock.release()


def make_snapshot(type):
	"""Generate snapshot after detection"""
	snapshot.generate(snapframes, [
		type + _(" LOGIN"),
		_("Date: ") + datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC"),
		_("Scan time: ") + str(round(time.time() - timings["fr"], 2)) + "s",
		_("Frames: ") + str(frames) + " (" + str(round(frames / (time.time() - timings["fr"]), 2)) + "FPS)",
		_("Hostname: ") + os.uname().nodename,
		_("Best certainty value: ") + str(round(lowest_certainty * 10, 1))
	])


def send_to_ui(type, message):
	"""Send message to the auth ui"""
	global gtk_proc

	# Only execute of the process started
	if "gtk_proc" in globals():
		# Format message so the ui can parse it
		message = type + "=" + message + " \n"

		# Try to send the message to the auth ui, but it's okay if that fails
		try:
			if gtk_proc.poll() is None: # Make sure the gtk_proc is still running before write into the pipe
				gtk_proc.stdin.write(bytearray(message.encode("utf-8")))
				gtk_proc.stdin.flush()
		except IOError:
			pass


# Make sure we were given an username to test against
if len(sys.argv) < 2:
	exit(12)

# The username of the user being authenticated
user = sys.argv[1]
# The model file contents
models = []
# Encoded face models
encodings = []
# Amount of ignored 100% black frames
black_tries = 0
# Amount of ignored dark frames
dark_tries = 0
# Total amount of frames captured
frames = 0
# Captured frames for snapshot capture
snapframes = []
# Tracks the lowest certainty value in the loop
lowest_certainty = 10
# Face recognition/detection instances
face_detector = None
pose_predictor = None
pose_predictor_68 = None
face_encoder = None

# Liveness detection state
blinks = 0
blink_counter = 0
blink_state = False
ear_history = []
liveness_start_time = None
consecutive_low_ear_frames = 0
last_blink_time = 0

# Try to load the face model from the models folder
try:
	models = json.load(open(paths_factory.user_model_path(user)))

	for model in models:
		encodings += model["data"]
except FileNotFoundError:
	exit(10)

# Check if the file contains a model
if len(models) < 1:
	exit(10)

# Read config from disk
config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

# Get all config values needed
use_cnn = config.getboolean("core", "use_cnn", fallback=False)
timeout = config.getint("video", "timeout", fallback=4)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=50.0)
video_certainty = config.getfloat("video", "certainty", fallback=3.5) / 10
end_report = config.getboolean("debug", "end_report", fallback=False)
save_failed = config.getboolean("snapshots", "save_failed", fallback=False)
save_successful = config.getboolean("snapshots", "save_successful", fallback=False)
gtk_stdout = config.getboolean("debug", "gtk_stdout", fallback=False)
rotate = config.getint("video", "rotate", fallback=0)

# Liveness detection config
liveness_enabled = config.getboolean("liveness", "enabled", fallback=True)

# Debug log file
debug_log = open("/tmp/howdy_debug.log", "a")
debug_log.write(f"[{datetime.now()}] Starting compare.py, liveness_enabled={liveness_enabled}\n")
debug_log.flush()

liveness_blinks_required = config.getint("liveness", "required_blinks", fallback=1)
liveness_timeout = config.getfloat("liveness", "liveness_timeout", fallback=3.0)
ear_threshold = config.getfloat("liveness", "ear_threshold", fallback=0.18)
blink_frames = config.getint("liveness", "blink_frames", fallback=4)
require_both_eyes = config.getboolean("liveness", "require_both_eyes", fallback=True)
check_eye_recovery = config.getboolean("liveness", "check_eye_recovery", fallback=True)
min_blink_interval = config.getfloat("liveness", "min_blink_interval", fallback=0.5)
ear_smoothing_window = config.getint("liveness", "ear_smoothing_window", fallback=5)
motion_threshold = config.getint("liveness", "motion_threshold", fallback=3)
ear_stability_threshold = config.getfloat("liveness", "ear_stability_threshold", fallback=0.03)

# Initialize blink detector with config
blink_detector.motion_threshold = motion_threshold
blink_detector.ear_stability_threshold = ear_stability_threshold

# IR camera validation config
ir_enforce = config.getboolean("video", "ir_enforce", fallback=False)
ir_warn = config.getboolean("video", "ir_warn", fallback=True)

# Send the gtk output to the terminal if enabled in the config
gtk_pipe = sys.stdout if gtk_stdout else subprocess.DEVNULL

# Start the auth ui, register it to be always be closed on exit
try:
	gtk_proc = subprocess.Popen(["howdy-gtk", "--start-auth-ui"], stdin=subprocess.PIPE, stdout=gtk_pipe, stderr=gtk_pipe)
	atexit.register(exit)
except FileNotFoundError:
	pass

# Write to the stdin to redraw ui
send_to_ui("M", _("Starting up..."))

# Save the time needed to start the script
timings["in"] = time.time() - timings["st"]

# Import face recognition, takes some time
timings["ll"] = time.time()

# Start threading and wait for init to finish
lock = thread.allocate_lock()
lock.acquire()
thread.start_new_thread(init_detector, (lock, ))

# Start video capture on the IR camera
timings["ic"] = time.time()

video_capture = VideoCapture(config)

# Read exposure from config to use in the main loop
exposure = config.getint("video", "exposure", fallback=-1)

# Note the time it took to open the camera
timings["ic"] = time.time() - timings["ic"]

# wait for thread to finish
lock.acquire()
lock.release()
del lock

# IR camera detection and validation
ir_info = None
if ir_enforce or ir_warn:
	is_ir, ir_info = detect_ir_camera(video_capture)
	if ir_info:
		if ir_enforce and not is_ir:
			print(_("IR camera enforcement enabled but no IR camera detected"))
			print(_("Camera stats: mean_brightness={:.1f}, std_brightness={:.1f}").format(
				ir_info["mean_brightness"], ir_info["std_brightness"]))
			exit(14)
		elif ir_warn and not is_ir:
			print(_("WARNING: No IR camera detected. Face recognition may be vulnerable to photo attacks."))
			print(_("Camera stats: mean_brightness={:.1f}, std_brightness={:.1f}").format(
				ir_info["mean_brightness"], ir_info["std_brightness"]))

# Fetch the max frame height
max_height = config.getfloat("video", "max_height", fallback=320.0)

# Get the height of the image (which would be the width if screen is portrait oriented)
height = video_capture.internal.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
if rotate == 2:
	height = video_capture.internal.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
# Calculate the amount the image has to shrink
scaling_factor = (max_height / height) or 1

# Fetch config settings out of the loop
timeout = config.getint("video", "timeout", fallback=4)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)
end_report = config.getboolean("debug", "end_report", fallback=False)

# Initiate histogram equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Let the ui know that we're ready
send_to_ui("M", _("Identifying you..."))

# Start the read loop
frames = 0
valid_frames = 0
timings["fr"] = time.time()
dark_running_total = 0

while True:
	# Increment the frame count every loop
	frames += 1

	# Form a string to let the user know we're real busy
	ui_subtext = "Scanned " + str(valid_frames - dark_tries) + " frames"
	if (dark_tries > 1):
		ui_subtext += " (skipped " + str(dark_tries) + " dark frames)"
	# Show it in the ui as subtext
	send_to_ui("S", ui_subtext)

	# Stop if we've exceeded the time limit
	if time.time() - timings["fr"] > timeout:
		# Create a timeout snapshot if enabled
		if save_failed:
			make_snapshot(_("FAILED"))

		if dark_tries == valid_frames:
			print(_("All frames were too dark, please check dark_threshold in config"))
			print(_("Average darkness: {avg}, Threshold: {threshold}").format(avg=str(dark_running_total / max(1, valid_frames)), threshold=str(dark_threshold)))
			exit(13)
		else:
			exit(11)

	# Grab a single frame of video
	frame, gsframe = video_capture.read_frame()
	gsframe = clahe.apply(gsframe)

	# If snapshots have been turned on
	if save_failed or save_successful:
		# Start capturing frames for the snapshot
		if len(snapframes) < 3:
			snapframes.append(frame)

	# Create a histogram of the image with 8 values
	hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
	# All values combined for percentage calculation
	hist_total = np.sum(hist)

	# Calculate frame darkness
	darkness = (hist[0] / hist_total * 100)

	# If the image is fully black due to a bad camera read,
	# skip to the next frame
	if (hist_total == 0) or (darkness == 100):
		black_tries += 1
		continue

	dark_running_total += darkness
	valid_frames += 1

	# If the image exceeds darkness threshold due to subject distance,
	# skip to the next frame
	if (darkness > dark_threshold):
		dark_tries += 1
		continue

	# If the height is too high
	if scaling_factor != 1:
		# Apply that factor to the frame
		frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
		gsframe = cv2.resize(gsframe, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

	# If camera is configured to rotate = 1, check portrait in addition to landscape
	if rotate == 1:
		if frames % 3 == 1:
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
		if frames % 3 == 2:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_CLOCKWISE)

	# If camera is configured to rotate = 2, check portrait orientation
	elif rotate == 2:
		if frames % 2 == 0:
			frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
		else:
			frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
			gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_CLOCKWISE)

	# Get all faces from that frame as encodings
	# Upsamples 1 time
	face_locations = face_detector(gsframe, 1)
	# Loop through each face
	for fl in face_locations:
		if use_cnn:
			fl = fl.rect

		# Fetch the faces in the image
		face_landmark = pose_predictor(frame, fl)
		face_encoding = np.array(face_encoder.compute_face_descriptor(frame, face_landmark, 1))

		# Liveness detection via blink detection
		# Skip liveness check entirely if no blinks required
		if liveness_enabled and liveness_blinks_required > 0 and pose_predictor_68 is not None:
			try:
				if liveness_start_time is None:
					liveness_start_time = time.time()
				
				liveness_elapsed = time.time() - liveness_start_time
				if liveness_elapsed > liveness_timeout:
					if end_report:
						print(_("Liveness check timeout after {:.1f}s").format(liveness_elapsed))
					exit(15)
				
				face_landmark_68 = pose_predictor_68(frame, fl)
				blinks, ear = blink_detector.detect(face_landmark_68, ear_threshold, blink_frames, require_both_eyes, check_eye_recovery, min_blink_interval, ear_smoothing_window)
				
				debug_log.write(f"[{datetime.now()}] Blinks: {blinks}/{liveness_blinks_required}, EAR: {ear:.3f}\n")
				debug_log.flush()
				
				# We no longer 'continue' here, we let the face recognition run in parallel
				# The check for blinks is now moved inside the match confirmation
			except Exception as e:
				if end_report:
					print(_("Liveness detection error: {}").format(e))
				continue

		# Match this found face against a known face
		matches = np.linalg.norm(encodings - face_encoding, axis=1)

		# Get best match
		match_index = np.argmin(matches)
		match = matches[match_index]

		# Update certainty if we have a new low
		if lowest_certainty > match:
			lowest_certainty = match

		# Check if a match that's confident enough
		if 0 < match < video_certainty:
			# If liveness is enabled, we only proceed if blinks are met
			if liveness_enabled and liveness_blinks_required > 0:
				if blinks < liveness_blinks_required:
					continue

			timings["tt"] = time.time() - timings["st"]
			timings["fl"] = time.time() - timings["fr"]

			# If set to true in the config, print debug text
			if end_report:
				def print_timing(label, k):
					"""Helper function to print a timing from the list"""
					print("  %s: %dms" % (label, round(timings[k] * 1000)))

				# Print a nice timing report
				print(_("Time spent"))
				print_timing(_("Starting up"), "in")
				print(_("  Open cam + load libs: %dms") % (round(max(timings["ll"], timings["ic"]) * 1000, )))
				print_timing(_("  Opening the camera"), "ic")
				print_timing(_("  Importing recognition libs"), "ll")
				print_timing(_("Searching for known face"), "fl")
				print_timing(_("Total time"), "tt")

				print(_("\nResolution"))
				width = video_capture.fw or 1
				print(_("  Native: %dx%d") % (height, width))
				# Save the new size for diagnostics
				scale_height, scale_width = frame.shape[:2]
				print(_("  Used: %dx%d") % (scale_height, scale_width))

				# Show the total number of frames and calculate the FPS by dividing it by the total scan time
				print(_("\nFrames searched: %d (%.2f fps)") % (frames, frames / timings["fl"]))
				print(_("Black frames ignored: %d ") % (black_tries, ))
				print(_("Dark frames ignored: %d ") % (dark_tries, ))
				print(_("Certainty of winning frame: %.3f") % (match * 10, ))

				print(_("Winning model: %d (\"%s\")") % (match_index, models[match_index]["label"]))

			# Make snapshot if enabled
			if save_successful:
				make_snapshot(_("SUCCESSFUL"))

			# Run rubberstamps if enabled
			if config.getboolean("rubberstamps", "enabled", fallback=False):
				import rubberstamps

				send_to_ui("S", "")

				if "gtk_proc" not in vars():
					gtk_proc = None

				rubberstamps.execute(config, gtk_proc, {
					"video_capture": video_capture,
					"face_detector": face_detector,
					"pose_predictor": pose_predictor,
					"clahe": clahe
				})

			# End peacefully
			exit(0)

	if exposure != -1:
		# For a strange reason on some cameras (e.g. Lenoxo X1E) setting manual exposure works only after a couple frames
		# are captured and even after a delay it does not always work. Setting exposure at every frame is reliable though.
		video_capture.internal.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)  # 1 = Manual
		video_capture.internal.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
