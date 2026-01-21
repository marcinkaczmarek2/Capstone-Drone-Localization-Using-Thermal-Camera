import airsim
import time
import numpy as np
import cv2
import csv
import os
import signal
import sys

# =======================
# KONFIGURACJA
# =======================
vehicles = ["Chaser", "Target"]
fps = 30
duration = 20  # czas lotu w sekundach
frame_size = (640, 480)

video_files = {v: f"{v.lower()}_video.mp4" for v in vehicles}
telemetry_file = "telemetry.csv"
output_dir = os.path.expanduser("~/Documents/AirSim/recordings")
os.makedirs(output_dir, exist_ok=True)

# Kamery przypisane w Settings.json (CameraID 0)
drone_cameras = {v: "0" for v in vehicles}

# =======================
# POŁĄCZENIE Z AIRSIM
# =======================
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim!")

# =======================
# FUNKCJE BEZPIECZNEGO ZAKOŃCZENIA
# =======================
def safe_exit(signum=None, frame=None):
    print("\n[INFO] Exiting safely...")
    try:
        for v_name in vehicles:
            client.landAsync(vehicle_name=v_name).join()
            client.armDisarm(False, vehicle_name=v_name)
            client.enableApiControl(False, vehicle_name=v_name)
    except Exception as e:
        print(f"[ERROR] During safe exit: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, safe_exit)

# =======================
# ARM I TAKEOFF
# =======================
for v_name in vehicles:
    client.enableApiControl(True, vehicle_name=v_name)
    client.armDisarm(True, vehicle_name=v_name)
    client.takeoffAsync(vehicle_name=v_name).join()

time.sleep(3)
print("Drones airborne!")

# =======================
# PRZYGOTOWANIE VIDEO
# =======================
video_writers = {}
for v in vehicles:
    path = os.path.join(output_dir, video_files[v])
    if os.path.exists(path):
        os.remove(path)
    video_writers[v] = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size
    )

# =======================
# PRZYGOTOWANIE TELEMETRII
# =======================
telemetry_path = os.path.join(output_dir, telemetry_file)
csv_file = open(telemetry_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    "VehicleName", "TimeStamp", "POS_X", "POS_Y", "POS_Z",
    "Q_W", "Q_X", "Q_Y", "Q_Z"
])

# =======================
# START RUCHU DRONÓW
# =======================
client.moveByVelocityAsync(-5, 10, -10, duration, vehicle_name="Target")
client.rotateByYawRateAsync(20, duration, vehicle_name="Chaser")

# =======================
# NAGRYWANIE I ZBIERANIE TELEMETRII
# =======================
start_time = time.time()
frame_count = 0
total_frames = int(duration * fps)

while frame_count < total_frames:
    now = time.time()
    elapsed = now - start_time
    timestamp = int(now * 1000)

    for v in vehicles:
        raw_image = client.simGetImage(
            camera_name=drone_cameras[v],
            image_type=airsim.ImageType.Scene,
            vehicle_name=v
        )

        if raw_image is None:
            print(f"[Warning] No image received from {v} at elapsed {elapsed:.2f}s")
            continue

        jpg_image = np.frombuffer(raw_image, dtype=np.uint8)
        img_bgr = cv2.imdecode(jpg_image, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[Warning] Failed to decode image from {v} at elapsed {elapsed:.2f}s")
            continue

        if img_bgr.shape[:2] != frame_size[::-1]:
            img_bgr = cv2.resize(img_bgr, frame_size)

        video_writers[v].write(img_bgr)

    # telemetria
    for v in vehicles:
        state = client.getMultirotorState(vehicle_name=v)
        pos = state.kinematics_estimated.position
        ori = state.kinematics_estimated.orientation
        csv_writer.writerow([
            v, timestamp, pos.x_val, pos.y_val, pos.z_val,
            ori.w_val, ori.x_val, ori.y_val, ori.z_val
        ])

    frame_count += 1
    if frame_count % fps == 0:
        print(f"[INFO] Elapsed: {int(elapsed)}s, frames recorded: {frame_count}")

    # synchronizacja z FPS
    next_frame_time = start_time + (frame_count / fps)
    sleep_time = max(0, next_frame_time - time.time())
    time.sleep(sleep_time)

# =======================
# ZAKOŃCZENIE VIDEO I TELEMETRII
# =======================
for v in vehicles:
    video_writers[v].release()
csv_file.close()
print(f"Videos and telemetry saved in: {output_dir}")

# =======================
# LĄDOWANIE I ROZBRAJANIE
# =======================
safe_exit()
