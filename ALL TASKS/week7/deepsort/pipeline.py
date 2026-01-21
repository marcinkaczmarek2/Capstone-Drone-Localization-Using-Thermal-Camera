# pipeline.py
import os
import argparse

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

        # Jesteś w katalogu YOLOv5-TEST, który ma hubconf.py,
        # więc można załadować model lokalnie przez torch.hub
        self.model = torch.hub.load(
            '.',           # aktualny katalog - YOLOv5-TEST
            'custom',      # typ modelu
            path=weights_path,
            source='local' # bardzo ważne: użyj lokalnego repo, nie z GitHuba
        ).to(self.device)

        self.model.eval()

    def detect(self, frame_bgr, conf_thres=0.4):
        """
        Zwraca ndarray N x 6:
        [x1, y1, x2, y2, conf, cls]
        gdzie cls = 0 (drone) lub 1 (notdrone)
        """
        # YOLOv5 oczekuje obrazu w RGB
        frame_rgb = frame_bgr[:, :, ::-1]

        # inference
        results = self.model(frame_rgb, size=640)
        # [x1, y1, x2, y2, conf, cls]
        detections = results.xyxy[0].detach().cpu().numpy()

        out = []
        for x1, y1, x2, y2, conf, cls in detections:
            if conf < conf_thres:
                continue

            # ZACHOWUJEMY OBU KLASY: 0 = drone, 1 = notdrone
            out.append([
                float(x1), float(y1), float(x2), float(y2),
                float(conf), int(cls)
            ])

        if len(out) == 0:
            return None

        return np.array(out)


def draw_box(frame, x1, y1, x2, y2, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label is not None:
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_pipeline(weights_path, source_path, output_path,
                 mode='deepsort', conf_thres=0.4, device='cuda'):
    """
    weights_path – ścieżka do best.pt
    source_path – ścieżka do pliku wideo (mp4, avi itp.)
    output_path – gdzie zapisać wynik
    mode – na razie 'deepsort'; można dodać 'yolo_only'
    """

    # --- 1. Przygotowanie wideo ---
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie mogę otworzyć źródła: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # --- 2. Inicjalizacja YOLOv5 i DeepSORT ---
    detector = DroneYoloV5Detector(weights_path, device=device)
    tracker = DeepSortTracker()

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- 3. Detekcja YOLOv5 ---
        detections = detector.detect(frame, conf_thres=conf_thres)

        if mode == 'yolo_only':
            # rysujemy detekcje z podziałem na klasy: drone / notdrone
            if detections is not None:
                for x1, y1, x2, y2, conf, cls in detections:
                    if int(cls) == 0:
                        label = f"drone {conf:.2f}"
                        color = (0, 255, 0)  # zielony dla drona
                    else:
                        label = f"notdrone {conf:.2f}"
                        color = (0, 0, 255)  # czerwony dla notdrone

                    draw_box(frame, x1, y1, x2, y2, color, label)



        elif mode == 'deepsort':
            drone_dets = None
            if detections is not None:
                drone_list = []
                for x1, y1, x2, y2, conf, cls in detections:
                    if int(cls) == 0:  # tylko drony
                        drone_list.append([x1, y1, x2, y2, conf])

                if len(drone_list) > 0:
                    drone_dets = np.array(drone_list)

            tracks = tracker.update(drone_dets, frame)

            for x1, y1, x2, y2, track_id, conf in tracks:
                label = f"ID {track_id}"
                draw_box(frame, x1, y1, x2, y2, (255, 0, 0), label)

        # --- 5. Zapis klatki do wideo wynikowego ---
        out.write(frame)

        # Jeśli chcesz podgląd na żywo, odkomentuj:
        # cv2.imshow('tracking', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Zapisano wynik do: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    # TU PODAJESZ ŚCIEŻKĘ DO SWOJEGO MODELU (DOMYŚLNIE TWÓJ best.pt)
    parser.add_argument(
        '--weights',
        type=str,
        default=r"runs\train\notdroneplusdrone_15_epochs\weights\best.pt",
        help="Ścieżka do wytrenowanego modelu YOLOv5 (best.pt)"
    )

    # ŚCIEŻKA DO ŹRÓDŁA – w Twoim przypadku np. plik wideo w siwobaza\
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help="Ścieżka do pliku wideo (np. siwobaza\\video1.mp4)"
    )

    # GDZIE ZAPISAĆ WYNIK
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
        help="Tryb działania: deepsort lub yolo_only (bez trackera)"
    )

    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.4,
        help="Próg confidence dla YOLOv5"
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help="cuda lub cpu"
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
        device=args.device
    )
