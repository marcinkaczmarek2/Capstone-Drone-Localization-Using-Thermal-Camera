# pipeline.py
import os
import argparse
import csv

import cv2
import torch
import numpy as np

from tracker_deepsort import DeepSortTracker


class DroneYoloV5Detector:
    def __init__(self, weights_path, device='cuda'):
        """
        weights_path – ścieżka do best.pt
        device – 'cuda' lub 'cpu'
        """
        if device == 'cuda' and not torch.cuda.is_available():
            print("Uwaga: CUDA niedostępne, przełączam na CPU")
            device = 'cpu'

        self.device = device

        # Lokalny YOLOv5 przez torch.hub
        self.model = torch.hub.load(
            '.',           # katalog z hubconf.py (Twój lokalny YOLOv5-TEST)
            'custom',
            path=weights_path,
            source='local'
        ).to(self.device)

        self.model.eval()

    def detect(self, frame_bgr, conf_thres=0.3):
        """
        Zwraca ndarray N x 6:
        [x1, y1, x2, y2, conf, cls]
        """
        frame_rgb = frame_bgr[:, :, ::-1]
        results = self.model(frame_rgb, size=640)
        detections = results.xyxy[0].detach().cpu().numpy()

        out = []
        for x1, y1, x2, y2, conf, cls in detections:
            if conf < conf_thres:
                continue
            out.append([float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)])

        if len(out) == 0:
            return None
        return np.array(out)


def draw_box(frame, x1, y1, x2, y2, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label is not None:
        cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_pipeline(weights_path, source_path, output_path,
                 mode='deepsort', conf_thres=0.3, device='cuda',
                 log_csv_path=None):
    """
    mode:
      - 'deepsort' : YOLO + DeepSORT
      - 'yolo_only': tylko detekcje YOLO
    log_csv_path: jeśli podasz ścieżkę, zapisze log tracków do CSV
    """

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie mogę otworzyć źródła: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    detector = DroneYoloV5Detector(weights_path, device=device)
    tracker = DeepSortTracker()

    csv_file = None
    csv_writer = None
    if log_csv_path:
        os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
        csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx", "track_id", "x1", "y1", "x2", "y2", "conf", "is_pred"])

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detections = detector.detect(frame, conf_thres=conf_thres)

            if mode == 'yolo_only':
                if detections is not None:
                    for x1, y1, x2, y2, conf, cls in detections:
                        if int(cls) == 0:
                            label = f"drone {conf:.2f}"
                            color = (0, 255, 0)
                        else:
                            label = f"notdrone {conf:.2f}"
                            color = (0, 0, 255)
                        draw_box(frame, x1, y1, x2, y2, color, label)

            elif mode == 'deepsort':
                drone_dets = None
                if detections is not None:
                    drone_list = []
                    for x1, y1, x2, y2, conf, cls in detections:
                        if int(cls) == 0:  # tylko dron
                            drone_list.append([x1, y1, x2, y2, conf])
                    if len(drone_list) > 0:
                        drone_dets = np.array(drone_list)

                tracks = tracker.update(drone_dets, frame)

                # Teraz tracker zwraca: [x1,y1,x2,y2, track_id, conf, is_pred]
                for tr in tracks:
                    if len(tr) == 6:
                        # kompatybilność gdybyś odpalił na starej wersji trackera
                        x1, y1, x2, y2, track_id, conf = tr
                        is_pred = 0
                    else:
                        x1, y1, x2, y2, track_id, conf, is_pred = tr

                    label = f"ID {track_id}" + (" (pred)" if int(is_pred) == 1 else "")
                    color = (0, 255, 255) if int(is_pred) == 1 else (255, 0, 0)
                    draw_box(frame, x1, y1, x2, y2, color, label)

                    if csv_writer is not None:
                        csv_writer.writerow([frame_idx, track_id, x1, y1, x2, y2, conf, int(is_pred)])

            out.write(frame)

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()

    print(f"Zapisano wynik do: {output_path}")
    if log_csv_path:
        print(f"Zapisano log tracków do: {log_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--weights',
        type=str,
        default=r"runs\train\notdroneplusdrone_15_epochs\weights\best.pt",
        help="Ścieżka do wytrenowanego modelu YOLOv5 (best.pt)"
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Ścieżka do pliku wideo (np. siwobaza\\video1.mp4)"
    )

    parser.add_argument(
        '--output',
        type=str,
        default=r"results\deepsort_output.mp4",
        help="Ścieżka do pliku wyjściowego z trackingiem"
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='deepsort',
        choices=['deepsort', 'yolo_only'],
        help="Tryb działania: deepsort lub yolo_only"
    )

    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.3,
        help="Próg confidence dla YOLOv5"
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="cuda lub cpu"
    )

    parser.add_argument(
        '--log-csv',
        type=str,
        default=None,
        help="Opcjonalnie: ścieżka do CSV z logiem tracków"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        weights_path=args.weights,
        source_path=args.source,
        output_path=args.output,
        mode=args.mode,
        conf_thres=args.conf_thres,
        device=args.device,
        log_csv_path=args.log_csv
    )
