import airsim
import time
import numpy as np
import cv2
import csv
import os
import signal
import sys
from airsim import Pose, Vector3r, to_quaternion

# =======================
# KONFIGURACJA
# =======================
FPS = 30
DURATION = 20  # sekundy
FRAME_SIZE = (640, 512)
CHASER = "Chaser"
TARGET = "Target"

OUTPUT_DIR = os.path.expanduser("~/Documents/AirSim/recordings")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_PATH = os.path.join(OUTPUT_DIR, "chaser_thermal.mp4")
TELEMETRY_PATH = os.path.join(OUTPUT_DIR, "telemetry.csv")
CAMERA_NAME = "thermal_cam"

# =======================
# THERMAL POST-PROCESS
# =======================
def thermal_preprocess(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# =======================
# BEZPIECZNE WYJŚCIE
# =======================
def safe_exit(signum=None, frame=None):
    print("\n[INFO] Exiting safely...")
    try:
        for v in [CHASER, TARGET]:
            client.landAsync(vehicle_name=v).join()
            client.armDisarm(False, vehicle_name=v)
            client.enableApiControl(False, vehicle_name=v)
    except Exception as e:
        print(f"[ERROR] Safe exit: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, safe_exit)

# =======================
# POŁĄCZENIE Z AIRSIM
# =======================
client = airsim.MultirotorClient()
client.confirmConnection()
print("[OK] Connected to AirSim")

# =======================
# ARM + TAKEOFF
# =======================
for v in [CHASER, TARGET]:
    client.enableApiControl(True, vehicle_name=v)
    client.armDisarm(True, vehicle_name=v)
    client.takeoffAsync(vehicle_name=v).join()
time.sleep(2)
print("[OK] Drones airborne")

# =======================
# VIDEO WRITER
# =======================
if os.path.exists(VIDEO_PATH):
    os.remove(VIDEO_PATH)

video_writer = cv2.VideoWriter(
    VIDEO_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    FPS,
    FRAME_SIZE
)

# =======================
# TELEMETRIA
# =======================
csv_file = open(TELEMETRY_PATH, mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "Vehicle", "Timestamp", "POS_X", "POS_Y", "POS_Z",
    "VEL_X", "VEL_Y", "VEL_Z", "Q_W", "Q_X", "Q_Y", "Q_Z"
])

# =======================
# PARAMETRY RUCHU
# =======================
TARGET_SPEED_XY = 3.0  # max prędkość w XY
TARGET_SPEED_Z = 1.0   # max prędkość w Z
CHASER_ALTITUDE_SPEED = 1.5  # prędkość wznoszenia chasera
DESIRED_ALTITUDE = 10.0      # docelowa wysokość chasera

# =======================
# NAGRYWANIE
# =======================
start_time = time.time()
frame_count = 0
total_frames = int(DURATION * FPS)
next_motion_time = time.time()

print("[REC] Recording started")

while frame_count < total_frames:
    now = time.time()
    timestamp = int(now * 1000)

    # =======================
    # LOSOWY RUCH TARGETU (co 1s)
    # =======================
    if now >= next_motion_time:
        vx = np.random.uniform(-TARGET_SPEED_XY, TARGET_SPEED_XY)
        vy = np.random.uniform(-TARGET_SPEED_XY, TARGET_SPEED_XY)
        vz = np.random.uniform(-TARGET_SPEED_Z, TARGET_SPEED_Z)
        yaw_target = np.random.uniform(-60, 60)
        client.moveByVelocityAsync(vx, vy, vz, 1, vehicle_name=TARGET)
        client.rotateByYawRateAsync(yaw_target, 1, vehicle_name=TARGET)
        next_motion_time = now + 1

    # =======================
    # POZYCJE DRONÓW
    # =======================
    state_target = client.getMultirotorState(vehicle_name=TARGET)
    pos_target = np.array([
        state_target.kinematics_estimated.position.x_val,
        state_target.kinematics_estimated.position.y_val,
        state_target.kinematics_estimated.position.z_val
    ])

    state_chaser = client.getMultirotorState(vehicle_name=CHASER)
    pos_chaser = np.array([
        state_chaser.kinematics_estimated.position.x_val,
        state_chaser.kinematics_estimated.position.y_val,
        state_chaser.kinematics_estimated.position.z_val
    ])

   # =======================
    # WZNOSZENIE CHASERA TYLKO W Z
    # =======================
    dz = DESIRED_ALTITUDE - pos_chaser[2]
    vz = np.clip(dz, -CHASER_ALTITUDE_SPEED, CHASER_ALTITUDE_SPEED)
    vz = -vz
    vz /= 4
    client.moveByVelocityAsync(0, 0, vz, 1, vehicle_name=CHASER)

    # =======================
    # Powolna rotacja chasera
    # =======================
    yaw_rate = 360 / 4  # 90°/s
    client.rotateByYawRateAsync(yaw_rate, 1, vehicle_name=CHASER)

    # =======================
    # OBRAZ Z CHASERA
    # =======================
    raw_image = client.simGetImage(CAMERA_NAME, airsim.ImageType.Scene, vehicle_name=CHASER)
    if raw_image is not None:
        jpg = np.frombuffer(raw_image, dtype=np.uint8)
        img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
        if img is not None:
            if img.shape[:2] != FRAME_SIZE[::-1]:
                img = cv2.resize(img, FRAME_SIZE)
            img = thermal_preprocess(img)
            video_writer.write(img)

    # =======================
    # TELEMETRIA
    # =======================
    for v in [CHASER, TARGET]:
        state = client.getMultirotorState(vehicle_name=v)
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        ori = state.kinematics_estimated.orientation
        csv_writer.writerow([
            v, timestamp,
            pos.x_val, pos.y_val, pos.z_val,
            vel.x_val, vel.y_val, vel.z_val,
            ori.w_val, ori.x_val, ori.y_val, ori.z_val
        ])

    frame_count += 1
    if frame_count % FPS == 0:
        print(f"[INFO] {frame_count // FPS}s recorded")

    # =======================
    # synchronizacja FPS
    # =======================
    next_frame = start_time + frame_count / FPS
    time.sleep(max(0, next_frame - time.time()))

# =======================
# ZAMYKANIE
# =======================
video_writer.release()
csv_file.close()
print(f"[DONE] Data saved to {OUTPUT_DIR}")
safe_exit()
