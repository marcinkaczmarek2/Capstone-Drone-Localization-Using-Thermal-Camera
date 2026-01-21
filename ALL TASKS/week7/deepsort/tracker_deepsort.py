# tracker_deepsort.py
from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    def __init__(self,
                 max_age=60,
                 n_init=5,
                 max_cosine_distance=0.2,
                 nn_budget=100):
        """
        max_age – ile klatek można 'nie widzieć' obiektu zanim track zostanie usunięty
        n_init – ile kolejnych detekcji potrzeba, żeby track był 'potwierdzony'

        Uwaga:
        DeepSORT wewnętrznie używa filtru Kalmana do przewidywania ruchu
        i kojarzenia detekcji między klatkami, ale w update() poniżej
        zwracamy TYLKO takie tracki, które zostały zaktualizowane detekcją
        YOLO w bieżącej klatce (time_since_update == 0). Dzięki temu
        na wyjściu nie ma "przewidywanych" prostokątów bez detekcji.
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

    def update(self, detections, frame_bgr):
        """
        detections: ndarray N x 5 [x1, y1, x2, y2, conf]
        frame_bgr: aktualna klatka (BGR z OpenCV)

        Zwraca listę tracków:
        [x1, y1, x2, y2, track_id, conf]

        WAŻNE:
        Zwracamy tylko tracki z time_since_update == 0, czyli takie,
        które w TEJ klatce były skojarzone z detekcją YOLO.
        Tracki oparte wyłącznie na predykcji Kalmana (bez nowej detekcji)
        są pomijane – nie rysujesz wtedy prostokąta.
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
                    'drone'  # etykieta klasy (tutaj umownie 'drone')
                ))

        # aktualizacja trackera
        tracks = self.tracker.update_tracks(deep_sort_dets, frame=frame_bgr)

        output_tracks = []
        for t in tracks:
            # Track musi być potwierdzony
            if not t.is_confirmed():
                continue

            # >>> KLUCZOWA LINIJKA <<<
            # time_since_update == 0 -> track został ZAKTUALIZOWANY
            # detekcją z bieżącej klatki (czyli była detekcja YOLO)
            # time_since_update > 0  -> to tylko predykcja Kalmana
            # bez nowej detekcji -> NIE rysujemy takiego tracka
            if t.time_since_update > 0:
                continue

            track_id = t.track_id
            # bbox w formacie [left, top, right, bottom]
            x1, y1, x2, y2 = t.to_ltrb()
            # pewność detekcji z ostatniej klatki (jak brak – ustawiamy 1.0)
            conf = t.det_conf if t.det_conf is not None else 1.0
            output_tracks.append([x1, y1, x2, y2, track_id, conf])

        return output_tracks
