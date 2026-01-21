import os, json

labels_dir = "dataset_test/YoloLabels/"
output_file = "predictions_yolo.json"

data = []

for file in os.listdir(labels_dir):
    if not file.endswith(".txt"):
        continue

    image_name = file.replace(".txt", ".jpg")
    with open(os.path.join(labels_dir, file)) as f:
        boxes = [list(map(float, line.split()[1:])) for line in f.readlines()]

    data.append({"image": image_name, "bboxes": boxes})

with open(output_file, "w") as f:
    json.dump(data, f, indent=2)

print(" Zapisano predictions_yolo.json")