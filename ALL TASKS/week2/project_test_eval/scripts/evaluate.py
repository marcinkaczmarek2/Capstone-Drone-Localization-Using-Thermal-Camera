import json
import numpy as np

IOU_THRESHOLDS = [0.5, 0.75]  # Możesz dodać więcej progów IOU

def iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    xi1, yi1 = max(x1_min, x2_min), max(y1_min, y2_min)
    xi2, yi2 = min(x1_max, x2_max), min(y1_max, y2_max)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    return inter / (area1 + area2 - inter + 1e-6)

# Wczytanie danych
gt = json.load(open("ground_truth.json"))
pred = json.load(open("predictions_yolo.json"))

for IOU_THRESHOLD in IOU_THRESHOLDS:
    TP = FP = FN = 0

    for g in gt:
        p = next((x for x in pred if x["image"] == g["image"]), {"bboxes": []})

        gt_boxes = g["bboxes"]
        pred_boxes = p["bboxes"]

        used = set()

        for gb in gt_boxes:
            matched = False
            for i, pb in enumerate(pred_boxes):
                if i in used:
                    continue
                if iou(gb, pb) >= IOU_THRESHOLD:
                    TP += 1
                    used.add(i)
                    matched = True
                    break
            if not matched:
                FN += 1

        FP += len(pred_boxes) - len(used)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print(f"\n=== Wyniki dla IOU ≥ {IOU_THRESHOLD} ===")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")
