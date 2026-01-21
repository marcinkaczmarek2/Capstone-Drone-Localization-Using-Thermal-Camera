# tracker_deepsort.py
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    def __init__(self,
                 max_age=60,
                 n_init=3,
                 max_cosine_distance=0.2,
                 nn_budget=100):
        """
        max_age – ile klatek można 'nie widzieć' obiektu zanim track zostanie usunięty
        n_init – ile kolejnych detekcji potrzeba, żeby track był 'potwierdzony'

        DeepSORT ma Kalman filter (predykcja ruchu). Jeśli w danej klatce
        nie ma dopasowanej detekcji, tracker nadal potrafi zwrócić predykcję.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    def update(self, detections, frame_bgr):
        """
        detections: ndarray N x 5 [x1, y1, x2, y2, conf] albo None
        frame_bgr: aktualna klatka (BGR)

        Zwraca listę tracków:
        [x1, y1, x2, y2, track_id, conf, is_pred]

        is_pred = 0 -> track zaktualizowany detekcją w tej klatce (time_since_update == 0)
        is_pred = 1 -> track jest tylko predykcją (brak dopasowanej detekcji)
        """
        deep_sort_dets = []

        # DeepSORT (deep-sort-realtime) oczekuje bboxów w formacie [x, y, w, h]
        if detections is not None:
            for det in detections:
                x1, y1, x2, y2, conf = det
                w = x2 - x1
                h = y2 - y1
                deep_sort_dets.append((
                    [float(x1), float(y1), float(w), float(h)],
                    float(conf),
                    'drone'
                ))

        tracks = self.tracker.update_tracks(deep_sort_dets, frame=frame_bgr)

        output_tracks = []
        for t in tracks:
            if not t.is_confirmed():
                continue

            track_id = t.track_id
            is_pred = 1 if t.time_since_update > 0 else 0

            x1, y1, x2, y2 = t.to_ltrb()
            conf = t.det_conf if t.det_conf is not None else 1.0

            output_tracks.append([x1, y1, x2, y2, track_id, conf, is_pred])

        return output_tracks
